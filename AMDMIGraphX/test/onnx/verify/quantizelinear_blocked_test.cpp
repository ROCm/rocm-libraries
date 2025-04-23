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

static migraphx::shape make_shape(migraphx::shape::type_t type, std::vector<size_t> lens)
{
    return migraphx::shape{type, std::move(lens)};
}

TEST_CASE(quantizelinear_2d_blocked_with_zp_test)
{
    migraphx::program p = read_onnx("quantizelinear_2d_blocked_with_zp_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {6, 2}};
    std::vector<float> x = {
        6.0f, 12.0f, 50.0f, 5.0f, 40.0f, 1.0f, 8.0f, 4.0f, 7.0f, 3.0f, 0.0f, 20.0f};

    migraphx::shape scale_shape{migraphx::shape::float_type, {2, 2}};
    std::vector<float> scale = {1.5f, 2.5f, 3.0f, 4.9f};

    migraphx::shape zp_shape{migraphx::shape::int8_type, {2, 2}};
    std::vector<int8_t> zp = {0, 1, 2, 3};

    migraphx::parameter_map pm;
    pm["x"]     = migraphx::argument{x_shape, x.data()};
    pm["scale"] = migraphx::argument{scale_shape, scale.data()};
    pm["zp"]    = migraphx::argument{zp_shape, zp.data()};

    auto result = p.eval(pm).back();
    EXPECT(result.get_shape() == make_shape(migraphx::shape::int8_type, {6, 2}));

    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {4, 6, 33, 3, 27, 1, 5, 4, 4, 4, 2, 7};
    EXPECT(result_vector == gold);
}

TEST_CASE(quantizelinear_2d_blocked_runt_block_test)
{
    migraphx::program p = read_onnx("quantizelinear_2d_blocked_runt_block_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {3, 5}};
    std::vector<float> x = {6.0f,
                            12.0f,
                            50.0f,
                            5.0f,
                            50.0f,
                            1.0f,
                            8.0f,
                            4.0f,
                            5.0f,
                            4.0f,
                            0.0f,
                            20.0f,
                            10.0f,
                            4.0f,
                            10.0f};

    migraphx::shape scale_shape{migraphx::shape::float_type, {3, 3}};
    std::vector<float> scale = {1.5f, 2.5f, 3.6f, 3.0f, 4.9f, 5.4f, 5.1f, 6.9f, 3.8f};

    migraphx::parameter_map pm;
    pm["x"]       = migraphx::argument{x_shape, x.data()};
    pm["y_scale"] = migraphx::argument{scale_shape, scale.data()};

    auto result = p.eval(pm).back();
    EXPECT(result.get_shape() == make_shape(migraphx::shape::uint8_type, {3, 5}));

    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<uint8_t> gold = {4, 8, 20, 2, 14, 0, 3, 1, 1, 1, 0, 4, 1, 1, 3};
    EXPECT(result_vector == gold);
}

TEST_CASE(quantizelinear_3d_blocked_with_zp_runt_block_test)
{
    migraphx::program p = read_onnx("quantizelinear_3d_blocked_with_zp_runt_block_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 5, 2}};
    std::vector<float> x = {6.0f, 1.0f,  12.0f, 8.0f,  50.0f, 4.0f,  5.0f,  7.0f,  40.0f, 3.0f,
                            0.0f, 14.0f, 20.0f, 42.0f, 13.0f, 25.0f, 17.0f, 22.0f, 31.0f, 33.0f};

    migraphx::shape scale_shape{migraphx::shape::float_type, {2, 2, 2}};
    std::vector<float> scale = {1.5f, 2.5f, 3.0f, 4.9f, 1.8f, 3.6f, 2.3f, 4.1f};

    migraphx::shape zp_shape{migraphx::shape::int8_type, {2, 2, 2}};
    std::vector<int8_t> zp = {0, 1, 2, 3, 3, 2, 1, 0};

    migraphx::parameter_map pm;
    pm["x"]     = migraphx::argument{x_shape, x.data()};
    pm["scale"] = migraphx::argument{scale_shape, scale.data()};
    pm["zp"]    = migraphx::argument{zp_shape, zp.data()};

    auto result = p.eval(pm).back();
    EXPECT(result.get_shape() == make_shape(migraphx::shape::int8_type, {2, 5, 2}));

    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {4, 1, 8, 4, 33, 3, 4, 4, 15, 4, 3, 6, 14, 14, 10, 9, 8, 5, 14, 8};
    EXPECT(result_vector == gold);
}
