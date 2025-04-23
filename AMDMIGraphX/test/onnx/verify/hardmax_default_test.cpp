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

TEST_CASE(hardmax_default_verify_test)
{
    migraphx::program p = read_onnx("hardmax_default_test.onnx");
    p.compile(migraphx::make_target("ref"));

    std::vector<std::size_t> input_lens{1, 2, 3, 4};
    auto input_type = migraphx::shape::float_type;
    migraphx::shape data_shape{input_type, input_lens};
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(data_shape, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                               0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
