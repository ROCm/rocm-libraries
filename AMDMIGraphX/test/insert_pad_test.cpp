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
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/normalize_ops.hpp>
#include <migraphx/insert_pad.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/common.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(
        m, {migraphx::normalize_ops{}, migraphx::insert_pad{}, migraphx::dead_code_elimination{}});
}

static migraphx::instruction_ref
create_im2col(migraphx::instruction_ref& l_img, size_t channels, migraphx::module& m)
{
    size_t f[2] = {1, 1};
    std::vector<int32_t> weights(channels * f[0] * f[1]);
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_weights = m.add_literal(migraphx::literal{s_weights, weights});
    return m.add_instruction(
        migraphx::make_op("im2col", {{"padding", {0, 0, 1, 1}}}), l_img, l_weights);
}

static migraphx::instruction_ref
create_conv(migraphx::instruction_ref& l_img,
            size_t channels,
            migraphx::module& m,
            migraphx::op::padding_mode_t padding_mode = migraphx::op::padding_mode_t::default_)
{
    migraphx::shape s_weights{migraphx::shape::int32_type, {4, channels, 3, 3}};
    std::vector<int32_t> weights(4 * channels * 3 * 3);
    auto l_weights = m.add_literal(migraphx::literal{s_weights, weights});
    return m.add_instruction(
        migraphx::make_op("convolution",
                          {{"padding_mode", padding_mode}, {"padding", {0, 0, 1, 1}}}),
        l_img,
        l_weights);
}

TEST_CASE(rewrite_pad)
{
    migraphx::module m;
    size_t img_dim[2] = {2, 2};
    size_t channels   = 1;
    std::vector<int32_t> input(channels * img_dim[0] * img_dim[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::shape s_img{migraphx::shape::int32_type, {1, channels, img_dim[0], img_dim[1]}};
    auto l_img = m.add_literal(migraphx::literal{s_img, input});

    auto l0 = create_im2col(l_img, channels, m);
    auto l1 = create_conv(l_img, channels, m);
    auto l2 = m.add_instruction(
        migraphx::make_op("pooling",
                          {{"mode", migraphx::op::pooling_mode::max}, {"padding", {0, 0, 1, 1}}}),
        l_img);
    m.add_instruction(migraphx::make_op("identity"), l0, l1, l2);

    run_pass(m);

    EXPECT(std::any_of(
        m.begin(), m.end(), [](const migraphx::instruction& ins) { return ins.name() == "pad"; }));
}

TEST_CASE(rewrite_pad_symmetric)
{
    migraphx::module m;

    size_t img_dim[2] = {2, 2};
    size_t channels   = 1;
    std::vector<int32_t> input(channels * img_dim[0] * img_dim[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::shape s_img{migraphx::shape::int32_type, {1, channels, img_dim[0], img_dim[1]}};
    auto l_img = m.add_literal(migraphx::literal{s_img, input});

    m.add_instruction(
        migraphx::make_op("pooling",
                          {{"mode", migraphx::op::pooling_mode::max}, {"padding", {1, 1, 1, 1}}}),
        l_img);

    run_pass(m);
    EXPECT(std::none_of(
        m.begin(), m.end(), [](const migraphx::instruction& ins) { return ins.name() == "pad"; }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
