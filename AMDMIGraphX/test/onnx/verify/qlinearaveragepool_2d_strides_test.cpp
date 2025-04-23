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

TEST_CASE(qlinearaveragepool_2d_strides_test)
{
    auto p = read_onnx("qlinearaveragepool_2d_strides_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<int8_t> data_x = {
        84,   -73,  117, -2,   -97,  72,  67,   27,  1,   -44,  110, 51,   9,    7,    58,  113,
        -34,  34,   124, -20,  6,    66,  68,   98,  31,  -84,  25,  101,  -69,  -100, -68, 116,
        33,   -121, 78,  49,   102,  -86, 65,   69,  -87, -89,  16,  -125, 51,   -54,  -86, 79,
        -112, -37,  -6,  74,   118,  -75, -41,  52,  101, -22,  -28, -92,  -59,  -128, 32,  78,
        -20,  121,  11,  -107, -92,  -31, 81,   117, -55, -3,   80,  119,  126,  -98,  -11, 52,
        -4,   -66,  37,  -57,  -16,  -33, -12,  100, 55,  2,    27,  62,   -15,  64,   -74, -21,
        -123, 22,   -45, 12,   30,   24,  20,   120, -36, -102, -75, -39,  -76,  55,   74,  -120,
        103,  67,   -80, -89,  -112, 36,  69,   98,  110, -82,  60,  119,  98,   88,   5,   42,
        -88,  -86,  -58, -33,  93,   80,  -57,  -56, 87,  7,    -4,  114,  -73,  -91,  -12, -123,
        96,   -99,  -31, -99,  85,   34,  -126, 106, 88,  126,  -60, 14,   75,   -117, -15, 6,
        55,   -14,  117, -87,  -75,  -50, -85,  54,  70,  125,  74,  -100, 25,   -112, 74,  -66,
        -116, -102, 1,   -75,  -107, 83,  -120, -66, 57,  29,   62,  -45,  -103, -56,  90,  -53};
    migraphx::shape s_x{migraphx::shape::int8_type, {1, 3, 8, 8}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {24, 37, 10, 17, 12, 12, -13, -1, 14, -10, 7, -19};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
