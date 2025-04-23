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

TEST_CASE(mean_broadcast_test)
{
    migraphx::program p = read_onnx("mean_broadcast_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s0{migraphx::shape::float_type, {1, 3, 4}};
    std::vector<float> data0(12, 1);
    migraphx::shape s1{migraphx::shape::float_type, {1, 2, 3, 4}};
    std::vector<float> data1(24, 2);
    migraphx::shape s2{migraphx::shape::float_type, {4}};
    std::vector<float> data2(4, 3);
    migraphx::shape s3{migraphx::shape::float_type, {1}};
    std::vector<float> data3(1, 4);
    migraphx::shape s4{migraphx::shape::float_type, {2, 3, 1}};
    std::vector<float> data4(6, 5);

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(s0, data0.data());
    pp["1"] = migraphx::argument(s1, data1.data());
    pp["2"] = migraphx::argument(s2, data2.data());
    pp["3"] = migraphx::argument(s3, data3.data());
    pp["4"] = migraphx::argument(s4, data4.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold(24, 3);
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
