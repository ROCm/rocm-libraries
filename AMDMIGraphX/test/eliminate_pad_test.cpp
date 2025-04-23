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
#include <migraphx/eliminate_pad.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <basic_ops.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(
        m,
        {migraphx::normalize_ops{}, migraphx::eliminate_pad{}, migraphx::dead_code_elimination{}});
}

static migraphx::instruction_ref
create_im2col(migraphx::instruction_ref& l_img, size_t channels, migraphx::module& m)
{
    size_t f[2] = {1, 1};
    std::vector<int32_t> weights(channels * f[0] * f[1]);
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_weights = m.add_literal(migraphx::literal{s_weights, weights});
    return m.add_instruction(migraphx::make_op("im2col"), l_img, l_weights);
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
        migraphx::make_op("convolution", {{"padding_mode", padding_mode}}), l_img, l_weights);
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
    auto padded_img =
        m.add_instruction(migraphx::make_op("pad", {{"pads", {0, 0, 1, 1, 0, 0, 1, 1}}}), l_img);

    auto l0 = create_im2col(padded_img, channels, m);
    auto l1 = create_conv(padded_img, channels, m);
    auto l2 = m.add_instruction(
        migraphx::make_op("pooling", {{"mode", migraphx::op::pooling_mode::max}}), padded_img);
    m.add_instruction(migraphx::make_op("identity"), l0, l1, l2);

    auto s0 = l0->get_shape();
    auto s1 = l1->get_shape();
    auto s2 = l2->get_shape();
    run_pass(m);
    EXPECT(l0->get_shape() == s0);
    EXPECT(l1->get_shape() == s1);
    EXPECT(l2->get_shape() == s2);
    auto op0 = l0->get_operator().to_value();
    auto om1 = l1->get_operator().to_value();
    auto om2 = l2->get_operator().to_value();

    EXPECT(op0["padding"].to_vector<std::size_t>() == std::vector<std::size_t>{1, 1, 1, 1});
    EXPECT(om1["padding"].to_vector<std::size_t>() == std::vector<std::size_t>{1, 1, 1, 1});
    EXPECT(om2["padding"].to_vector<std::size_t>() == std::vector<std::size_t>{1, 1, 1, 1});

    EXPECT(std::none_of(
        m.begin(), m.end(), [](const migraphx::instruction& ins) { return ins.name() == "pad"; }));
}

TEST_CASE(rewrite_pad_im2col_asymmetric)
{
    migraphx::module m;

    size_t img_dim[2] = {2, 2};
    size_t channels   = 1;
    std::vector<int32_t> input(channels * img_dim[0] * img_dim[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::shape s_img{migraphx::shape::int32_type, {1, channels, img_dim[0], img_dim[1]}};
    auto l_img = m.add_literal(migraphx::literal{s_img, input});
    auto padded_img =
        m.add_instruction(migraphx::make_op("pad", {{"pads", {0, 0, 0, 0, 0, 0, 2, 2}}}), l_img);

    auto l0 = create_im2col(padded_img, channels, m);

    auto s0 = l0->get_shape();
    run_pass(m);
    EXPECT(l0->get_shape() == s0);
    auto op0 = l0->get_operator().to_value();

    EXPECT(op0["padding"].to_vector<std::size_t>() == std::vector<std::size_t>{0, 0, 2, 2});

    run_pass(m);
    EXPECT(std::none_of(
        m.begin(), m.end(), [](const migraphx::instruction& ins) { return ins.name() == "pad"; }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
