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

TEST_CASE(qlinearaveragepool_2d_pads_count_include_pad_test)
{
    auto p = read_onnx("qlinearaveragepool_2d_pads_count_include_pad_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<int8_t> data_x = {-30,  50,  91,  -87,  -21, -113, -16, 6,    -128, 104,  82,  -126,
                                  54,   41,  -71, 62,   -11, -111, 13,  104,  -43,  -48,  30,  85,
                                  -62,  -33, -27, -114, 32,  -17,  30,  -26,  -18,  15,   17,  100,
                                  -122, 115, 84,  -34,  -86, 82,   102, -117, -91,  -105, 112, 91};
    migraphx::shape s_x{migraphx::shape::int8_type, {1, 3, 4, 4}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {
        15,   43,  94,  62,  34,  -16, 4,   -31, 10,  -6,  29,  -13, -67, -45,  43,   27,  4,   -83,
        -21,  -3,  -6,  15,  -3,  0,   -9,  71,  78,  83,  3,   -4,  62,  85,   45,   50,  27,  66,
        26,   -36, -29, 35,  97,  90,  2,   -86, -62, 73,  127, 127, -32, -128, -128, -24, 83,  74,
        -9,   -63, -45, -35, 20,  1,   15,  -12, -11, -72, -44, -46, 50,  40,   57,   25,  34,  18,
        22,   30,  40,  105, 97,  88,  -46, 26,  83,  127, 125, 69,  -94, 24,   127,  127, 116, 4,
        -128, -83, 83,  127, 127, -1,  -66, -79, 40,  124, 127, 18,  -19, -77,  -15,  86,  127, 83};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
