/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

TEST_CASE(gelu_quick_test)
{
    migraphx::program p = read_onnx("gelu_quick_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape shape{migraphx::shape::float_type, {3, 3}};
    std::vector<float> x = {0.0400717,
                            0.76666826,
                            0.75319463,
                            0.13215327,
                            0.37472633,
                            0.77117795,
                            0.95669776,
                            0.09139277,
                            0.37507972};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(shape, x.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // output from onnxruntime CPU EP
    std::vector<float> gold = {0.02023656,
                               0.45591995,
                               0.4466837,
                               0.0682589,
                               0.20486447,
                               0.45902082,
                               0.5906249,
                               0.04674028,
                               0.2050741};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
