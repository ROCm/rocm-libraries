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

TEST_CASE(round_half_test)
{
    migraphx::program p = read_onnx("round_half_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape xs{migraphx::shape::half_type, {4, 4}};
    std::vector<float> tmp = {-3.51,
                              -3.5,
                              -3.49,
                              -2.51,
                              -2.50,
                              -2.49,
                              -1.6,
                              -1.5,
                              -0.51,
                              -0.5,
                              0.5,
                              0.6,
                              2.4,
                              2.5,
                              3.5,
                              4.5};
    std::vector<migraphx::half> data{tmp.cbegin(), tmp.cend()};
    migraphx::parameter_map param_map;
    param_map["x"] = migraphx::argument(xs, data.data());

    auto result = p.eval(param_map).back();

    std::vector<migraphx::half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    tmp = {-4.0, -4.0, -3.0, -3.0, -2.0, -2.0, -2.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 4.0, 4.0};
    std::vector<migraphx::half> gold{tmp.cbegin(), tmp.cend()};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
