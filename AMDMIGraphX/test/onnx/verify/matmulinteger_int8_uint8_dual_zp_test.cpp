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

TEST_CASE(matmulinteger_int8_uint8_dual_zp_test)
{
    migraphx::program p = read_onnx("matmulinteger_int8_uint8_dual_zp_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s0{migraphx::shape::int8_type, {4, 3}};
    std::vector<int8_t> data0 = {-1, 5, -9, -2, 6, 10, -3, 7, -11, -4, 8, 0};
    migraphx::shape s1{migraphx::shape::uint8_type, {3, 2}};
    std::vector<uint8_t> data1 = {128, 129, 126, 131, 124, 133};

    migraphx::parameter_map pp;
    pp["1"] = migraphx::argument(s0, data0.data());
    pp["2"] = migraphx::argument(s1, data1.data());

    auto result = p.eval(pp).back();
    std::vector<int32_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<int32_t> gold = {-984, 992, 1351, -1362, -1234, 1244, 117, -118};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
