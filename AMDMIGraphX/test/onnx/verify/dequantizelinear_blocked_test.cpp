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
#include <onnx_verify_utils.hpp>

TEST_CASE(dequantizelinear_2d_blocked_with_zp_test)
{
    migraphx::program p = read_onnx("dequantizelinear_2d_blocked_with_zp_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::int8_type, {2, 2}};
    std::vector<int8_t> x = {4, 8, 20, 2};

    migraphx::shape scale_shape{migraphx::shape::float_type, {2, 2}};
    std::vector<float> scale = {1.5f, 2.5f, 3.0f, 4.9f};

    migraphx::shape zp_shape{migraphx::shape::int8_type, {2, 2}};
    std::vector<int8_t> zp = {0, 1, 2, 3};

    migraphx::parameter_map pm;
    pm["x"]     = migraphx::argument{x_shape, x.data()};
    pm["scale"] = migraphx::argument{scale_shape, scale.data()};
    pm["zp"]    = migraphx::argument{zp_shape, zp.data()};

    auto result = p.eval(pm).back();
    EXPECT(result.get_shape() == migraphx::shape{migraphx::shape::float_type, {2, 2}});

    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {6.0f, 17.5f, 54.0f, -4.9f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(dequantizelinear_3d_blocked_with_zp_runt_block_test)
{
    migraphx::program p = read_onnx("dequantizelinear_3d_blocked_with_zp_runt_block_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::int8_type, {2, 5, 2}};
    std::vector<int8_t> x = {4, 1, 8, 4, 33, 3, 4, 4, 15, 4, 3, 6, 14, 14, 10, 9, 8, 5, 14, 8};

    migraphx::shape scale_shape{migraphx::shape::float_type, {2, 2, 2}};
    std::vector<float> scale = {1.5f, 2.5f, 3.0f, 4.9f, 1.8f, 3.6f, 2.3f, 4.1f};

    migraphx::shape zp_shape{migraphx::shape::int8_type, {2, 2, 2}};
    std::vector<int8_t> zp = {0, 1, 2, 3, 3, 2, 1, 0};

    migraphx::parameter_map pm;
    pm["x"]     = migraphx::argument{x_shape, x.data()};
    pm["scale"] = migraphx::argument{scale_shape, scale.data()};
    pm["zp"]    = migraphx::argument{zp_shape, zp.data()};

    auto result = p.eval(pm).back();
    EXPECT(result.get_shape() == migraphx::shape{migraphx::shape::float_type, {2, 5, 2}});

    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {6.0f, 0.0f,  12.0f, 7.5f,  49.5f, 5.0f,  6.0f,  4.9f,  39.0f, 4.9f,
                               0.0f, 14.4f, 19.8f, 43.2f, 12.6f, 25.2f, 16.1f, 20.5f, 29.9f, 32.8f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
