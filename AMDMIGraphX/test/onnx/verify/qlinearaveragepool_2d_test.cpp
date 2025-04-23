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

TEST_CASE(qlinearaveragepool_2d_test)
{
    auto p = read_onnx("qlinearaveragepool_2d_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<int8_t> data_x = {84,  -73, 117, -2,  -97, 72,   67,  27,   1,  -44,  110, 51,
                                  9,   7,   58,  113, -34, 34,   124, -20,  6,  66,   68,  98,
                                  31,  -84, 25,  101, -69, -100, -68, 116,  33, -121, 78,  49,
                                  102, -86, 65,  69,  -87, -89,  16,  -125, 51, -54,  -86, 79};
    migraphx::shape s_x{migraphx::shape::int8_type, {1, 3, 4, 4}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {4,   127, 127, -41,  127, 127, -6,   125,  127,
                                76,  127, 127, 32,   78,  127, -128, -128, 127,
                                -44, -37, 127, -117, -62, 37,  -128, -128, -81};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
