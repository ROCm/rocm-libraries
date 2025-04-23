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

TEST_CASE(qlinearaveragepool_1d_test)
{
    auto p = read_onnx("qlinearaveragepool_1d_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<int8_t> data_x = {
        -31,  51,  125,  30,   -17,  -125, 121,  -19, -13,  52,   18,  -70,  97,   15,  56,   42,
        -65,  -26, 40,   -109, -70,  83,   110,  -94, 34,   70,   5,   -23,  -60,  -68, 19,   48,
        -113, 3,   -44,  20,   -99,  -103, -49,  -38, 122,  75,   38,  -7,   -65,  -56, 96,   99,
        50,   -27, -114, 49,   -65,  105,  -3,   54,  8,    38,   -81, -46,  -86,  -46, -104, 36,
        22,   -51, 48,   59,   -116, 6,    93,   16,  -111, 98,   51,  -87,  -111, -74, -39,  7,
        107,  115, 59,   60,   -66,  -14,  -106, -23, 119,  -122, -51, -100, 26,   125, 45,   90};
    migraphx::shape s_x{migraphx::shape::int8_type, {1, 3, 32}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {
        26,  104, 94,  22,  -55, 14,  67, 0,   36,  51,  -10, 29,  72,  52,  65,  5,
        -30, 23,  -19, -74, 23,  112, 24, -14, 68,  54,  7,   -26, -48, -8,  50,  -39,
        -4,  4,   -24, -85, -60, -28, 58, 114, 72,  31,  -20, -44, 36,  114, 90,  28,
        -54, -16, 8,   36,  67,  42,  47, 39,  -6,  -48, -50, -50, -59, -18, 2,   15,
        70,  -13, -39, 66,  71,  -32, 9,  90,  -2,  -83, -76, -40, 0,   73,  127, 103,
        75,  13,  -24, -44, -48, 64,  15, -70, -60, -21, 92,  101, 84};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
