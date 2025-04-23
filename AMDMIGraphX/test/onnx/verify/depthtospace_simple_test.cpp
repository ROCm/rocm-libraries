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

TEST_CASE(depthtospace_simple_test)
{
    auto p = read_onnx("depthtospace_simple_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<float> data_in(48);
    std::iota(std::begin(data_in), std::end(data_in), 0);
    migraphx::shape s_x{migraphx::shape::float_type, {1, 8, 2, 3}};
    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument(s_x, data_in.data());
    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0,  12, 1,  13, 2,  14, 24, 36, 25, 37, 26, 38, 3,  15, 4,  16,
                               5,  17, 27, 39, 28, 40, 29, 41, 6,  18, 7,  19, 8,  20, 30, 42,
                               31, 43, 32, 44, 9,  21, 10, 22, 11, 23, 33, 45, 34, 46, 35, 47};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
