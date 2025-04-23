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

TEST_CASE(resize_downsample_f_dyn2_test)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 10};

    auto p = read_onnx("resize_downsample_f_dyn2_test.onnx", options);
    p.compile(migraphx::make_target("ref"));

    // A Resize op. with static input shape goes through a different code path
    // but should give same result
    auto reference_p = read_onnx("resize_downsample_f_ref2_test.onnx", options);
    reference_p.compile(migraphx::make_target("ref"));

    migraphx::shape sx{migraphx::shape::float_type, {2, 1, 5, 9}};
    std::vector<float> dx(sx.elements());
    std::iota(dx.begin(), dx.end(), 0.1f);

    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sx, dx.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    // clang-format off
    // gold values different than in resize_downsample_f_dyn_test because the deduced scales are
    //   slightly different.
    std::vector<float> gold = {
         0.1f,  1.1f,  3.1f,  5.1f,  7.1f, 
         9.1f, 10.1f, 12.1f, 14.1f, 16.1f, 
        27.1f, 28.1f, 30.1f, 32.1f, 34.1f,
        45.1f, 46.1f, 48.1f, 50.1f, 52.1f, 
        54.1f, 55.1f, 57.1f, 59.1f, 61.1f, 
        72.1f, 73.1f, 75.1f, 77.1f, 79.1f
    };
    // clang-format on

    EXPECT(migraphx::verify::verify_range_with_tolerance(result_vector,
                                                         migraphx::verify::expected{gold}));

    auto reference_result = reference_p.eval(pp).back();
    std::vector<float> reference_vector;
    reference_result.visit(
        [&](auto output) { reference_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range_with_tolerance(
        result_vector, migraphx::verify::expected{reference_vector}));
}
