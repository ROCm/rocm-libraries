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
 *
 */

#include <migraphx/rewrite_dot.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::rewrite_dot{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(nchw_conv_1x1)
{
    migraphx::shape s1{migraphx::shape::float_type, {64, 128, 28, 28}};
    migraphx::shape s2{migraphx::shape::float_type, {512, 128, 1, 1}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s1);
        auto w    = m1.add_literal(migraphx::generate_literal(s2));
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x         = m2.add_parameter("x", s1);
        auto w         = m2.add_literal(migraphx::generate_literal(s2));
        auto squeeze   = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {2, 3}}}), w);
        auto broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {64, 512, 128}}}), squeeze);
        auto reshape1 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {64, 128, 784}}}), x);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), broadcast, reshape1);
        auto reshape2 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {64, 512, 28, 28}}}), dot);
        m2.add_return({reshape2});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_conv_1x1)
{
    auto s1 = migraphx::shape::from_permutation(
        migraphx::shape::float_type, {64, 128, 28, 28}, {0, 2, 3, 1});
    auto s2 = migraphx::shape::from_permutation(
        migraphx::shape::float_type, {512, 128, 1, 1}, {0, 2, 3, 1});
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s1);
        auto w    = m1.add_literal(migraphx::generate_literal(s2));
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x       = m2.add_parameter("x", s1);
        auto w       = m2.add_literal(migraphx::generate_literal(s2));
        auto squeeze = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {2, 3}}}), w);
        auto transpose1 =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), squeeze);
        auto broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {64, 28, 128, 512}}}), transpose1);
        auto transpose2 =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), x);
        auto dot        = m2.add_instruction(migraphx::make_op("dot"), transpose2, broadcast);
        auto transpose3 = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), dot);
        m2.add_return({transpose3});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_group_conv_1x1)
{
    auto s1 = migraphx::shape::from_permutation(
        migraphx::shape::float_type, {64, 192, 83, 83}, {0, 2, 3, 1});
    auto s2 = migraphx::shape::from_permutation(
        migraphx::shape::float_type, {84, 96, 1, 1}, {0, 2, 3, 1});
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s1);
        auto w    = m1.add_literal(migraphx::generate_literal(s2));
        auto conv = m1.add_instruction(migraphx::make_op("convolution", {{"group", 2}}), x, w);
        m1.add_return({conv});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
