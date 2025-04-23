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

TEST_CASE(gelu_default_half_test)
{
    migraphx::program p = read_onnx("gelu_default_half_test.onnx");
    p.compile(migraphx::make_target("ref"));

    std::vector<std::size_t> input_lens{3, 3};
    auto input_type = migraphx::shape::half_type;
    migraphx::shape data_shape{input_type, input_lens};
    std::vector<float> tmp = {-100.0f, -7.5f, -5.2f, -1.0f, 0.0f, 1.5f, 4.9f, 8.2f, 1000.0f};
    std::vector<migraphx::half> data = {tmp.begin(), tmp.end()};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(data_shape, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // gold values according to specification:
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-59
    // x = np.array([-100.0, -7.5, -5.2, -1.0, 0.0, 1.5, 4.9, 8.2, 1000.0]).astype(np.float16)
    // (0.5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))).astype(np.float16)
    tmp = {0.0f, 0.0f, -5.364e-07f, -0.1587f, 0.0f, 1.399f, 4.898f, 8.203f, 1000.0f};

    std::vector<migraphx::half> gold = {tmp.begin(), tmp.end()};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
