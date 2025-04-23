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

TEST_CASE(matmulintegertofloat_int8_uint8_scales_zp_bias_test)
{
    migraphx::program p = read_onnx("matmulintegertofloat_zp_bias_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s0{migraphx::shape::int8_type, {4, 3}};
    std::vector<int8_t> data0 = {-1, 5, -9, -2, 6, 10, -3, 7, -11, -4, 8, 0};
    migraphx::shape s1{migraphx::shape::uint8_type, {3, 2}};
    std::vector<uint8_t> data1 = {128, 129, 126, 131, 124, 133};

    migraphx::shape scale1_s{migraphx::shape::float_type, {3}};
    std::vector<float> scale1 = {1.0f, 1.0f, 1.0f};

    migraphx::shape scale2_s{migraphx::shape::float_type, {2}};
    std::vector<float> scale2 = {2.0f, 2.0f};

    migraphx::shape zp1_s{migraphx::shape::int8_type, {3}};
    std::vector<int8_t> zp1 = {1, 2, 3};

    migraphx::shape zp2_s{migraphx::shape::uint8_type, {2}};
    std::vector<uint8_t> zp2 = {3, 5};

    migraphx::shape bias_s{migraphx::shape::float_type, {2}};
    std::vector<float> bias = {-10.0f, -1.0f};

    migraphx::parameter_map pp;
    pp["1"] = migraphx::argument(s0, data0.data());
    pp["2"] = migraphx::argument(s1, data1.data());
    pp["3"] = migraphx::argument(scale1_s, scale1.data());
    pp["4"] = migraphx::argument(scale2_s, scale2.data());
    pp["5"] = migraphx::argument(zp1_s, zp1.data());
    pp["6"] = migraphx::argument(zp2_s, zp2.data());
    pp["7"] = migraphx::argument(bias_s, bias.data());

    auto result = p.eval(pp).back();
    std::vector<int32_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<int32_t> gold = {-2656, -2811, 1938, 2057, -3148, -3315, -490, -495};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
