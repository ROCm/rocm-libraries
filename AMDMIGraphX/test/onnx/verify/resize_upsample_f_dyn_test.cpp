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

TEST_CASE(resize_upsample_f_dyn_test)
{
    // resize with half_pixel and round_prefer_ceil, with scale > 1
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 10};

    auto p = read_onnx("resize_upsample_f_dyn_test.onnx", options);
    p.compile(migraphx::make_target("ref"));

    // should upscale to 2x4x8
    migraphx::shape sx{migraphx::shape::float_type, {2, 1, 3, 5}};
    std::vector<float> dx(sx.elements());
    std::iota(dx.begin(), dx.end(), 0.1f);

    migraphx::parameter_map pp;
    pp["X"]     = migraphx::argument(sx, dx.data());
    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // clang-format off
    std::vector<float> gold = { 
        0.1f,  0.1f,    1.1f,  2.1f,    2.1f,  3.1f,    4.1f,  4.1f,  
        0.1f,  0.1f,    1.1f,  2.1f,    2.1f,  3.1f,    4.1f,  4.1f,  
        5.1f,  5.1f,    6.1f,  7.1f,    7.1f,  8.1f,    9.1f,  9.1f,  
        10.1f,  10.1f,  11.1f,  12.1f,  12.1f,  13.1f,  14.1f,  14.1f,  
        15.1f,  15.1f,  16.1f,  17.1f,  17.1f,  18.1f,  19.1f,  19.1f,  
        15.1f,  15.1f,  16.1f,  17.1f,  17.1f,  18.1f,  19.1f,  19.1f,  
        20.1f,  20.1f,  21.1f,  22.1f,  22.1f,  23.1f,  24.1f,  24.1f,  
        25.1f,  25.1f,  26.1f,  27.1f,  27.1f,  28.1f,  29.1f,  29.1f};
    // clang-format on

    // Using verify_range_with_tolerance() because floating-point
    // rounding errors were observed.
    EXPECT(migraphx::verify::verify_range_with_tolerance(result_vector,
                                                         migraphx::verify::expected{gold}));
}
