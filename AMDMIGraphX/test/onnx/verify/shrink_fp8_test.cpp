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

#include <migraphx/float8.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(shrink_fp8_test)
{
    migraphx::program p = optimize_onnx("shrink_fp8_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::fp8e4m3fnuz_type, {3, 3}};
    // TODO: Make FP8 vector work for initializer list.
    std::vector<float> tmp_data{-4, -3, -2, -1, 0, 1, 2, 3, 4};
    std::vector<migraphx::fp8::fp8e4m3fnuz> data{tmp_data.cbegin(), tmp_data.cend()};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<migraphx::fp8::fp8e4m3fnuz> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    // TODO: Make FP8 vector work for initializer list.
    std::vector<float> tmp_gold = {-2, -1, 0, 0, 0, 0, 0, 1, 2};
    std::vector<migraphx::fp8::fp8e4m3fnuz> gold{tmp_gold.cbegin(), tmp_gold.cend()};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
