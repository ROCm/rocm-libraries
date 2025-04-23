/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(im2col_3x3_no_pad_identity_test)
{
    std::size_t f[2]    = {3, 3};
    std::size_t size[2] = {3, 3};
    std::vector<std::size_t> padding{0, 0};
    std::vector<std::size_t> stride{1, 1};
    std::vector<std::size_t> dilation{1, 1};
    std::size_t channels = 1;

    std::vector<int32_t> weights(channels * f[0] * f[1]);
    std::vector<int32_t> gold(channels * size[0] * size[1]);
    std::iota(gold.begin(), gold.end(), 0);

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s_image{migraphx::shape::int32_type, {1, channels, size[0], size[1]}};
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_image   = mm->add_literal(migraphx::literal{s_image, gold});
    auto l_weights = mm->add_literal(migraphx::literal{s_weights, weights});
    mm->add_instruction(
        migraphx::make_op("im2col",
                          {{"padding", padding}, {"stride", stride}, {"dilation", dilation}}),
        l_image,
        l_weights);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::size_t col_height = (size[0] - f[0] + 2 * padding[0]) / stride[0] + 1;
    std::size_t col_width  = (size[1] - f[1] + 2 * padding[1]) / stride[1] + 1;
    std::vector<float> results_vector(channels * f[0] * f[1] * col_height * col_width);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(im2col_3x3_no_pad_test)
{
    std::size_t f[2]    = {3, 3};
    std::size_t size[2] = {4, 4};
    std::vector<std::size_t> padding{0, 0};
    std::vector<std::size_t> stride{1, 1};
    std::vector<std::size_t> dilation{1, 1};
    std::size_t channels = 1;

    std::vector<int32_t> weights(channels * f[0] * f[1]);
    std::vector<int32_t> input(channels * size[0] * size[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s_image{migraphx::shape::int32_type, {1, channels, size[0], size[1]}};
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_image   = mm->add_literal(migraphx::literal{s_image, input});
    auto l_weights = mm->add_literal(migraphx::literal{s_weights, weights});
    mm->add_instruction(
        migraphx::make_op("im2col",
                          {{"padding", padding}, {"stride", stride}, {"dilation", dilation}}),
        l_image,
        l_weights);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<int> gold = {0, 1, 2, 4, 5, 6,  8,  9,  10, 1, 2, 3, 5, 6,  7,  9,  10, 11,
                             4, 5, 6, 8, 9, 10, 12, 13, 14, 5, 6, 7, 9, 10, 11, 13, 14, 15};

    std::size_t col_height = (size[0] - f[0] + 2 * padding[0]) / stride[0] + 1;
    std::size_t col_width  = (size[1] - f[1] + 2 * padding[1]) / stride[1] + 1;
    std::vector<float> results_vector(channels * f[0] * f[1] * col_height * col_width);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(im2col_3x3_stride_2_no_pad_test)
{
    std::size_t f[2]    = {3, 3};
    std::size_t size[2] = {6, 6};
    std::vector<std::size_t> padding{0, 0};
    std::vector<std::size_t> stride{2, 2};
    std::vector<std::size_t> dilation{1, 1};
    std::size_t channels = 1;

    std::vector<int32_t> weights(channels * f[0] * f[1]);
    std::vector<int32_t> input(channels * size[0] * size[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s_image{migraphx::shape::int32_type, {1, channels, size[0], size[1]}};
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_image   = mm->add_literal(migraphx::literal{s_image, input});
    auto l_weights = mm->add_literal(migraphx::literal{s_weights, weights});
    mm->add_instruction(
        migraphx::make_op("im2col",
                          {{"padding", padding}, {"stride", stride}, {"dilation", dilation}}),
        l_image,
        l_weights);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<int> gold = {0,  1,  2,  6,  7,  8,  12, 13, 14, 2,  3,  4,
                             8,  9,  10, 14, 15, 16, 12, 13, 14, 18, 19, 20,
                             24, 25, 26, 14, 15, 16, 20, 21, 22, 26, 27, 28};

    std::size_t col_height = (size[0] - f[0] + 2 * padding[0]) / stride[0] + 1;
    std::size_t col_width  = (size[1] - f[1] + 2 * padding[1]) / stride[1] + 1;
    std::vector<float> results_vector(channels * f[0] * f[1] * col_height * col_width);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(im2col_3x3_with_channels_identity_test)
{
    std::size_t f[2]    = {3, 3};
    std::size_t size[2] = {3, 3};
    std::vector<std::size_t> padding{0, 0};
    std::vector<std::size_t> stride{1, 1};
    std::vector<std::size_t> dilation{1, 1};
    std::size_t channels = 2;

    std::vector<int32_t> weights(channels * f[0] * f[1]);
    std::vector<int32_t> gold(channels * size[0] * size[1]);
    std::iota(gold.begin(), gold.end(), 0);

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s_image{migraphx::shape::int32_type, {1, channels, size[0], size[1]}};
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_image   = mm->add_literal(migraphx::literal{s_image, gold});
    auto l_weights = mm->add_literal(migraphx::literal{s_weights, weights});
    mm->add_instruction(
        migraphx::make_op("im2col",
                          {{"padding", padding}, {"stride", stride}, {"dilation", dilation}}),
        l_image,
        l_weights);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::size_t col_height = (size[0] - f[0] + 2 * padding[0]) / stride[0] + 1;
    std::size_t col_width  = (size[1] - f[1] + 2 * padding[1]) / stride[1] + 1;
    std::vector<float> results_vector(channels * f[0] * f[1] * col_height * col_width);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(im2col_3x3_with_padding_test)
{
    std::size_t f[2]    = {3, 3};
    std::size_t size[2] = {2, 2};
    std::vector<std::size_t> padding{1, 1};
    std::vector<std::size_t> stride{1, 1};
    std::vector<std::size_t> dilation{1, 1};
    std::size_t channels = 1;

    std::vector<int32_t> weights(channels * f[0] * f[1]);
    std::vector<int32_t> input(channels * size[0] * size[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s_image{migraphx::shape::int32_type, {1, channels, size[0], size[1]}};
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_image   = mm->add_literal(migraphx::literal{s_image, input});
    auto l_weights = mm->add_literal(migraphx::literal{s_weights, weights});
    mm->add_instruction(
        migraphx::make_op("im2col",
                          {{"padding", padding}, {"stride", stride}, {"dilation", dilation}}),
        l_image,
        l_weights);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<int> gold = {0, 0, 0, 0, 0, 1, 0, 2, 3, 0, 0, 0, 0, 1, 0, 2, 3, 0,
                             0, 0, 1, 0, 2, 3, 0, 0, 0, 0, 1, 0, 2, 3, 0, 0, 0, 0};

    std::size_t col_height = (size[0] - f[0] + 2 * padding[0]) / stride[0] + 1;
    std::size_t col_width  = (size[1] - f[1] + 2 * padding[1]) / stride[1] + 1;
    std::vector<float> results_vector(channels * f[0] * f[1] * col_height * col_width);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
