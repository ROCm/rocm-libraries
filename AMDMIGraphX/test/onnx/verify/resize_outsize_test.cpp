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

TEST_CASE(resize_outsize_test)
{
    // resize using output_size input, rather than scales
    migraphx::program p = read_onnx("resize_outsize_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    std::vector<float> dx(sx.elements());
    std::iota(dx.begin(), dx.end(), 0.1f);

    migraphx::shape sy{migraphx::shape::float_type, {1, 1, 4, 6}};
    std::vector<float> dy(sx.elements(), 0);

    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sx, dx.data());

    // Input Y is defined as type int64 in the Onnx file and will therefore be
    // interpreted as output shape (not scales) even though the input array is type float.
    pp["Y"] = migraphx::argument(sx, dy.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // clang-format off
    std::vector<float> gold = {0.1f, 0.1f, 1.1f, 1.1f, 1.1f, 1.1f, 
                                2.1f, 2.1f, 3.1f, 3.1f, 3.1f, 3.1f, 
                                2.1f, 2.1f, 3.1f, 3.1f, 3.1f, 3.1f, 
                                2.1f, 2.1f, 3.1f, 3.1f, 3.1f, 3.1f};
    // clang-format on

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
