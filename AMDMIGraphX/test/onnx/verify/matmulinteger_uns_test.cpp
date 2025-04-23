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

TEST_CASE(matmulinteger_uns_test)
{
    migraphx::program p = read_onnx("matmulinteger_uns_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s0{migraphx::shape::uint8_type, {4, 3}};
    std::vector<uint8_t> data0 = {11, 7, 3, 10, 6, 2, 9, 5, 1, 8, 4, 0};
    migraphx::shape s1{migraphx::shape::uint8_type, {3, 2}};
    std::vector<uint8_t> data1 = {1, 4, 2, 5, 3, 6};

    migraphx::parameter_map pp;
    pp["1"] = migraphx::argument(s0, data0.data());
    pp["2"] = migraphx::argument(s1, data1.data());

    auto result = p.eval(pp).back();
    std::vector<int32_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<int32_t> gold = {45730, 44641, 46108, 45010, 46486, 45379, 46864, 45748};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
