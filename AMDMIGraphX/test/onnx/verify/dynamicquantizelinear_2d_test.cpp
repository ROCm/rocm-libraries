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

TEST_CASE(dynamicquantizelinear_2d_test)
{
    auto p = read_onnx("dynamicquantizelinear_2d_test.onnx");
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{1.0, 2.1, 1.3, 2.5, 3.34, 4.0, 1.5, 2.6, 3.9, 4.0, 3.0, 2.345};
    migraphx::shape s_x{migraphx::shape::float_type, {3, 4}};
    migraphx::parameter_map pp;
    pp["x"]      = migraphx::argument(s_x, data.data());
    auto results = p.eval(pp);

    std::vector<uint8_t> y_results;
    results.at(0).visit([&](auto output) { y_results.assign(output.begin(), output.end()); });
    std::vector<uint8_t> y_gold = {64, 134, 83, 159, 213, 255, 96, 166, 249, 255, 191, 149};
    EXPECT(migraphx::verify::verify_rms_range(y_results, y_gold));

    std::vector<float> y_scale;
    results.at(1).visit([&](auto output) { y_scale.assign(output.begin(), output.end()); });
    std::vector<float> y_scale_gold = {0.0156862754};
    EXPECT(migraphx::verify::verify_rms_range(y_scale, y_scale_gold));

    std::vector<uint8_t> y_zpt;
    results.at(2).visit([&](auto output) { y_zpt.assign(output.begin(), output.end()); });
    std::vector<uint8_t> y_zpt_gold = {0};
    EXPECT(migraphx::verify::verify_rms_range(y_zpt, y_zpt_gold));
}
