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

TEST_CASE(mod_test_different_types)
{
    migraphx::program p = read_onnx("mod_test_different_dtypes.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s_int16{migraphx::shape::int16_type, {3, 3, 3}};
    migraphx::shape s_int32{migraphx::shape::int32_type, {3, 3, 3}};

    std::vector<int16_t> a = {-4, 7, 5,  4, -7, 8, -4, 7, 5,  4, -7, 8, -4, 7,
                              5,  4, -7, 8, -4, 7, 5,  4, -7, 8, -4, 7, 5};

    std::vector<int32_t> b = {2, -3, 8, -2, 3, 5,  2, -3, 8, -2, 3, 5,  2, -3,
                              8, -2, 3, 5,  2, -3, 8, -2, 3, 5,  2, -3, 8};

    migraphx::parameter_map p_map;
    p_map["0"] = migraphx::argument(s_int16, a.data());
    p_map["1"] = migraphx::argument(s_int32, b.data());

    auto result = p.eval(p_map).back();
    std::vector<int32_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int32_t> gold = {0, -2, 5, 0, 2, 3,  0, -2, 5, 0, 2, 3,  0, -2,
                                 5, 0,  2, 3, 0, -2, 5, 0,  2, 3, 0, -2, 5};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
