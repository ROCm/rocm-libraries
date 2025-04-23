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

TEST_CASE(hardmax_axis_neg_ver11_verify_test)
{
    migraphx::program p = read_onnx("hardmax_axis_neg_ver11_test.onnx");
    p.compile(migraphx::make_target("ref"));

    std::vector<std::size_t> input_lens{1, 2, 3, 4};
    auto input_type = migraphx::shape::half_type;
    migraphx::shape data_shape{input_type, input_lens};
    std::vector<float> tmp = {1.22461,   1.68262,   -2.0293,  -0.322021, 0.469971,  0.258057,
                              0.754395,  2.57422,   -1.68457, 0.0927734, 0.901855,  -0.876465,
                              -0.408936, 0.929688,  2.07227,  -1.57031,  0.486572,  -0.149292,
                              0.695312,  -0.217896, 0.713867, 0.717285,  0.0182953, 1.34961};

    std::vector<migraphx::half> data{tmp.cbegin(), tmp.cend()};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(data_shape, data.data());

    auto result = p.eval(pp).back();
    std::vector<migraphx::half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    tmp = {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<migraphx::half> gold{tmp.cbegin(), tmp.cend()};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
