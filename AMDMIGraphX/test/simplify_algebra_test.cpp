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
#include <migraphx/simplify_algebra.hpp>
#include <migraphx/fuse_pointwise.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>
#include <test.hpp>
#include <pointwise.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::simplify_algebra{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(simplify_add1)
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = m1.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = m1.add_literal(1);
        auto two  = m1.add_literal(2);
        auto sum1 = m1.add_instruction(migraphx::make_op("add"), x, one);
        auto sum2 = m1.add_instruction(migraphx::make_op("add"), y, two);
        auto sum3 = m1.add_instruction(migraphx::make_op("add"), sum1, sum2);
        m1.add_instruction(pass_op{}, sum3);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = m2.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = m2.add_literal(1);
        auto two  = m2.add_literal(2);
        auto sum1 = m2.add_instruction(migraphx::make_op("add"), one, two);
        auto sum2 = m2.add_instruction(migraphx::make_op("add"), x, y);
        auto sum3 = m2.add_instruction(migraphx::make_op("add"), sum2, sum1);
        m2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_add2)
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = m1.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = m1.add_literal(1);
        auto two  = m1.add_literal(2);
        auto sum1 = m1.add_instruction(migraphx::make_op("add"), one, x);
        auto sum2 = m1.add_instruction(migraphx::make_op("add"), two, y);
        auto sum3 = m1.add_instruction(migraphx::make_op("add"), sum1, sum2);
        m1.add_instruction(pass_op{}, sum3);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = m2.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = m2.add_literal(1);
        auto two  = m2.add_literal(2);
        auto sum1 = m2.add_instruction(migraphx::make_op("add"), one, two);
        auto sum2 = m2.add_instruction(migraphx::make_op("add"), x, y);
        auto sum3 = m2.add_instruction(migraphx::make_op("add"), sum2, sum1);
        m2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_add3)
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto one  = m1.add_literal(1);
        auto two  = m1.add_literal(2);
        auto sum1 = m1.add_instruction(migraphx::make_op("add"), one, x);
        auto sum2 = m1.add_instruction(migraphx::make_op("add"), one, two);
        auto sum3 = m1.add_instruction(migraphx::make_op("add"), sum1, sum2);
        m1.add_instruction(pass_op{}, sum3);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto one  = m2.add_literal(1);
        auto two  = m2.add_literal(2);
        auto sum1 = m2.add_instruction(migraphx::make_op("add"), one, two);
        auto sum2 = m2.add_instruction(migraphx::make_op("add"), one, sum1);
        auto sum3 = m2.add_instruction(migraphx::make_op("add"), x, sum2);
        m2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_zero_add_constant)
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto zero = m1.add_literal(0);
        m1.add_instruction(migraphx::make_op("add"), zero, x);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        m2.add_instruction(migraphx::make_op("identity"), x);
    }

    migraphx::module m3;
    {
        auto x    = m3.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto zero = m3.add_literal(0);
        m3.add_instruction(migraphx::make_op("add"), x, zero);
    }
    run_pass(m3);

    EXPECT((m1 == m2) and (m2 == m3));
}

TEST_CASE(simplify_add_broadcast1)
{
    migraphx::shape inner{migraphx::shape::int32_type, {2}};
    migraphx::shape outer{migraphx::shape::int32_type, {1, 2, 3, 3}};
    auto b = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 2, 3, 3}}});
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", outer);
        auto y    = m1.add_parameter("y", outer);
        auto one  = m1.add_literal({inner, {1, 1}});
        auto oneb = m1.add_instruction(b, one);
        auto two  = m1.add_literal({inner, {2, 2}});
        auto twob = m1.add_instruction(b, two);
        auto sum1 = m1.add_instruction(migraphx::make_op("add"), x, oneb);
        auto sum2 = m1.add_instruction(migraphx::make_op("add"), y, twob);
        auto sum3 = m1.add_instruction(migraphx::make_op("add"), sum1, sum2);
        m1.add_instruction(pass_op{}, sum3);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x     = m2.add_parameter("x", outer);
        auto y     = m2.add_parameter("y", outer);
        auto one   = m2.add_literal({inner, {1, 1}});
        auto two   = m2.add_literal({inner, {2, 2}});
        auto sum1  = m2.add_instruction(migraphx::make_op("add"), one, two);
        auto sum1b = m2.add_instruction(b, sum1);
        auto sum2  = m2.add_instruction(migraphx::make_op("add"), x, y);
        auto sum3  = m2.add_instruction(migraphx::make_op("add"), sum2, sum1b);
        m2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_add_broadcast2)
{
    migraphx::shape inner{migraphx::shape::int32_type, {2}};
    migraphx::shape outer{migraphx::shape::int32_type, {1, 2, 3, 3}};
    auto b              = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 2, 3, 3}}});
    auto create_program = [&] {
        migraphx::module m;
        auto x    = m.add_parameter("x", outer);
        auto y    = m.add_parameter("y", outer);
        auto one  = m.add_literal({inner, {1, 1}});
        auto oneb = m.add_instruction(b, one);
        auto two  = m.add_literal({outer, {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}});
        auto sum1 = m.add_instruction(migraphx::make_op("add"), x, y);
        auto sum2 = m.add_instruction(migraphx::make_op("add"), oneb, two);
        auto sum3 = m.add_instruction(migraphx::make_op("add"), sum2, sum1);
        m.add_instruction(pass_op{}, sum3);
        return m;
    };
    migraphx::module m1 = create_program();
    run_pass(m1);

    migraphx::module m2 = create_program();
    EXPECT(m1 == m2);
}

// TODO: Add test case
// TEST_CASE(simplify_add4)
[[maybe_unused]] static void simplify_add4()
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = m1.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = m1.add_literal(1);
        auto two  = m1.add_literal(2);
        auto sum1 = m1.add_instruction(migraphx::make_op("add"), one, x);
        auto sum2 = m1.add_instruction(migraphx::make_op("add"), sum1, y);
        auto sum3 = m1.add_instruction(migraphx::make_op("add"), sum2, two);
        m1.add_instruction(pass_op{}, sum3);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = m2.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = m2.add_literal(1);
        auto two  = m2.add_literal(2);
        auto sum1 = m2.add_instruction(migraphx::make_op("add"), one, two);
        auto sum2 = m2.add_instruction(migraphx::make_op("add"), x, y);
        auto sum3 = m2.add_instruction(migraphx::make_op("add"), sum2, sum1);
        m2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_mul_conv1)
{
    migraphx::module m;
    auto x = m.add_parameter("x", {migraphx::shape::int32_type, {1, 128, 28, 28}});
    auto w =
        m.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {256, 128, 3, 3}}));
    auto conv = m.add_instruction(
        migraphx::make_op("convolution",
                          {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
        x,
        w);
    auto a = m.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {256}}));
    auto b = m.add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 256, 14, 14}}}), a);
    auto mul = m.add_instruction(migraphx::make_op("mul"), conv, b);
    m.add_instruction(pass_op{}, mul);
    EXPECT(conv->outputs().front()->name() == "mul");
    run_pass(m);
    auto new_conv =
        std::find_if(m.begin(), m.end(), [](auto&& ins) { return ins.name() == "convolution"; });
    EXPECT(new_conv->outputs().front()->name() != "mul");
}

TEST_CASE(simplify_mul_conv2)
{
    migraphx::module m;
    auto x = m.add_parameter("x", {migraphx::shape::int32_type, {1, 128, 28, 28}});
    auto w =
        m.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {256, 128, 3, 3}}));
    auto conv = m.add_instruction(
        migraphx::make_op("convolution",
                          {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
        x,
        w);
    auto a      = m.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {256}}));
    auto unsq_a = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), a);
    auto b      = m.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, 256, 14, 14}}}), unsq_a);
    auto mul = m.add_instruction(migraphx::make_op("mul"), conv, b);
    m.add_instruction(pass_op{}, mul);
    EXPECT(conv->outputs().front()->name() == "mul");
    run_pass(m);
    auto new_conv =
        std::find_if(m.begin(), m.end(), [](auto&& ins) { return ins.name() == "convolution"; });
    EXPECT(new_conv->outputs().front()->name() != "mul");
}

// len = 1 case
TEST_CASE(simplify_mul_conv3)
{
    migraphx::module m;
    auto x = m.add_parameter("x", {migraphx::shape::int32_type, {1, 128, 28, 28}});
    auto w =
        m.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {256, 128, 3, 3}}));
    auto conv = m.add_instruction(
        migraphx::make_op("convolution",
                          {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
        x,
        w);
    auto a = m.add_literal(
        migraphx::generate_literal({migraphx::shape::int32_type, {256, 1, 1}, {1, 18, 1}}));
    auto b =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 256, 14, 14}}}), a);
    auto mul = m.add_instruction(migraphx::make_op("mul"), conv, b);
    m.add_instruction(pass_op{}, mul);
    EXPECT(conv->outputs().front()->name() == "mul");
    run_pass(m);
    auto new_conv =
        std::find_if(m.begin(), m.end(), [](auto&& ins) { return ins.name() == "convolution"; });
    EXPECT(new_conv->outputs().front()->name() != "mul");
}

// Previously broadcasted literal case, should skip
TEST_CASE(simplify_mul_conv_skip1)
{
    migraphx::module m;
    auto x = m.add_parameter("x", {migraphx::shape::int32_type, {1, 128, 28, 28}});
    auto w =
        m.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {256, 128, 3, 3}}));
    auto conv = m.add_instruction(
        migraphx::make_op("convolution",
                          {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
        x,
        w);
    auto a = m.add_literal(
        migraphx::generate_literal({migraphx::shape::int32_type, {256, 14, 14}, {1, 0, 0}}));
    auto b = m.add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 256, 14, 14}}}), a);
    auto mul = m.add_instruction(migraphx::make_op("mul"), conv, b);
    m.add_instruction(pass_op{}, mul);
    EXPECT(conv->outputs().front()->name() == "mul");
    run_pass(m);
    auto new_conv =
        std::find_if(m.begin(), m.end(), [](auto&& ins) { return ins.name() == "convolution"; });
    EXPECT(new_conv->outputs().front()->name() == "mul");
}

// Another previously broadcasted literal case, should skip
TEST_CASE(simplify_mul_conv_skip2)
{
    migraphx::module m;
    auto x = m.add_parameter("x", {migraphx::shape::int32_type, {1, 128, 28, 28}});
    auto w =
        m.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {256, 128, 3, 3}}));
    auto conv = m.add_instruction(
        migraphx::make_op("convolution",
                          {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
        x,
        w);
    auto a = m.add_literal(
        migraphx::generate_literal({migraphx::shape::int32_type, {256, 14, 14}, {1, 0, 0}}));
    auto b =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 256, 14, 14}}}), a);
    auto mul = m.add_instruction(migraphx::make_op("mul"), conv, b);
    m.add_instruction(pass_op{}, mul);
    EXPECT(conv->outputs().front()->name() == "mul");
    run_pass(m);
    auto new_conv =
        std::find_if(m.begin(), m.end(), [](auto&& ins) { return ins.name() == "convolution"; });
    EXPECT(new_conv->outputs().front()->name() == "mul");
}

TEST_CASE(simplify_mul_slice_conv1)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::int32_type, {1, 1024, 17, 17}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::int32_type, {768, 1024, 1, 1}}));
        auto conv   = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {384}}}), conv);
        auto a = m1.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {384}}));
        auto b = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 384, 17, 17}}}), a);
        auto mul    = m1.add_instruction(migraphx::make_op("mul"), slice1, b);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {384}}, {"ends", {768}}}), conv);
        auto add = m1.add_instruction(migraphx::make_op("add"), mul, slice2);
        m1.add_instruction(pass_op{}, add);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::int32_type, {1, 1024, 17, 17}});
        auto w = m2.add_literal(
            migraphx::generate_literal({migraphx::shape::int32_type, {768, 1024, 1, 1}}));
        auto wslice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {384}}}), w);
        auto a = m2.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {384}}));
        auto b = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {384, 1024, 1, 1}}}), a);
        auto mul     = m2.add_instruction(migraphx::make_op("mul"), b, wslice1);
        auto wslice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {384}}, {"ends", {768}}}), w);
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), mul, wslice2);
        auto conv   = m2.add_instruction(migraphx::make_op("convolution"), x, concat);
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {384}}}), conv);
        auto slice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {384}}, {"ends", {768}}}), conv);
        auto add = m2.add_instruction(migraphx::make_op("add"), slice1, slice2);
        m2.add_instruction(pass_op{}, add);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_mul_slice_conv_overlapping_slice)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::int32_type, {1, 1024, 17, 17}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::int32_type, {768, 1024, 1, 1}}));
        auto conv   = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {384}}}), conv);
        auto a = m1.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {384}}));
        auto b = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 384, 17, 17}}}), a);
        auto mul    = m1.add_instruction(migraphx::make_op("mul"), slice1, b);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {383}}, {"ends", {767}}}), conv);
        auto add = m1.add_instruction(migraphx::make_op("add"), mul, slice2);
        m1.add_instruction(pass_op{}, add);
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_mul_slice_conv_not_all_slice)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::int32_type, {1, 1024, 17, 17}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::int32_type, {768, 1024, 1, 1}}));
        auto conv   = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {384}}}), conv);
        auto a = m1.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {384}}));
        auto b = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 384, 17, 17}}}), a);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), slice1, b);
        auto c   = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::int32_type, {1, 768, 17, 17}}));
        auto add    = m1.add_instruction(migraphx::make_op("add"), conv, c);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), mul, add);
        m1.add_instruction(pass_op{}, concat);
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_mul_add)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto one = m1.add_literal(3);
        auto two = m1.add_literal(2);
        auto sum = m1.add_instruction(migraphx::make_op("add"), one, x);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), sum, two);
        m1.add_instruction(pass_op{}, mul);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto one  = m2.add_literal(3);
        auto two  = m2.add_literal(2);
        auto mul1 = m2.add_instruction(migraphx::make_op("mul"), two, x);
        auto mul2 = m2.add_instruction(migraphx::make_op("mul"), two, one);
        auto sum  = m2.add_instruction(migraphx::make_op("add"), mul1, mul2);
        m2.add_instruction(pass_op{}, sum);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_dot_add)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::float_type, {2, 2}});
        auto one = m1.add_literal(get_2x2());
        auto two = m1.add_literal(get_2x2(1));
        auto sum = m1.add_instruction(migraphx::make_op("add"), one, x);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), sum, two);
        m1.add_instruction(pass_op{}, dot);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", {migraphx::shape::float_type, {2, 2}});
        auto one  = m2.add_literal(get_2x2());
        auto two  = m2.add_literal(get_2x2(1));
        auto dot1 = m2.add_instruction(migraphx::make_op("dot"), x, two);
        auto dot2 = m2.add_instruction(migraphx::make_op("dot"), one, two);
        auto sum  = m2.add_instruction(migraphx::make_op("add"), dot1, dot2);
        m2.add_instruction(pass_op{}, sum);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_conv_add)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 32, 32}};
    migraphx::shape ws{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s);
        auto c    = m1.add_literal(migraphx::generate_literal(s, 1));
        auto w    = m1.add_literal(migraphx::generate_literal(ws, 2));
        auto sum  = m1.add_instruction(migraphx::make_op("add"), c, x);
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), sum, w);
        m1.add_instruction(pass_op{}, conv);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x     = m2.add_parameter("x", s);
        auto c     = m2.add_literal(migraphx::generate_literal(s, 1));
        auto w     = m2.add_literal(migraphx::generate_literal(ws, 2));
        auto conv1 = m2.add_instruction(migraphx::make_op("convolution"), c, w);
        auto conv2 = m2.add_instruction(migraphx::make_op("convolution"), x, w);
        auto sum   = m2.add_instruction(migraphx::make_op("add"), conv1, conv2);
        m2.add_instruction(pass_op{}, sum);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_inner_broadcast1)
{
    auto b = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 1, 4, 5}}});
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y   = m1.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto xb  = m1.add_instruction(b, x);
        auto yb  = m1.add_instruction(b, y);
        auto sum = m1.add_instruction(migraphx::make_op("add"), xb, yb);
        m1.add_instruction(pass_op{}, sum);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = m2.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto sum  = m2.add_instruction(migraphx::make_op("add"), x, y);
        auto sumb = m2.add_instruction(b, sum);
        m2.add_instruction(pass_op{}, sumb);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_inner_broadcast2)
{
    auto b = migraphx::make_op("multibroadcast", {{"out_lens", {2, 1, 4, 5}}});
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::int32_type, {1, 1, 1, 1}});
        auto y   = m1.add_parameter("y", {migraphx::shape::int32_type, {1, 1, 1, 1}});
        auto xb  = m1.add_instruction(b, x);
        auto yb  = m1.add_instruction(b, y);
        auto sum = m1.add_instruction(migraphx::make_op("add"), xb, yb);
        m1.add_instruction(pass_op{}, sum);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", {migraphx::shape::int32_type, {1, 1, 1, 1}});
        auto y    = m2.add_parameter("y", {migraphx::shape::int32_type, {1, 1, 1, 1}});
        auto sum  = m2.add_instruction(migraphx::make_op("add"), x, y);
        auto sumb = m2.add_instruction(b, sum);
        m2.add_instruction(pass_op{}, sumb);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_inner_broadcast_scalar)
{
    auto b = migraphx::make_op("multibroadcast", {{"out_lens", {32, 384}}});
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::int32_type, {1, 384}});
        auto y   = m1.add_parameter("y", {migraphx::shape::int32_type, {1, 1}});
        auto xb  = m1.add_instruction(b, x);
        auto yb  = m1.add_instruction(b, y);
        auto sum = m1.add_instruction(migraphx::make_op("add"), xb, yb);
        m1.add_instruction(pass_op{}, sum);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::int32_type, {1, 384}});
        auto y = m2.add_parameter("y", {migraphx::shape::int32_type, {1, 1}});
        auto yb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 384}}}), y);
        auto sum  = m2.add_instruction(migraphx::make_op("add"), x, yb);
        auto sumb = m2.add_instruction(b, sum);
        m2.add_instruction(pass_op{}, sumb);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_inner_broadcast_different_dims)
{
    auto b = migraphx::make_op("multibroadcast", {{"out_lens", {2, 384, 768}}});
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::int32_type, {384, 768}});
        auto y   = m1.add_parameter("y", {migraphx::shape::int32_type, {768}});
        auto xb  = m1.add_instruction(b, x);
        auto yb  = m1.add_instruction(b, y);
        auto sum = m1.add_instruction(migraphx::make_op("add"), xb, yb);
        m1.add_instruction(pass_op{}, sum);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", {migraphx::shape::int32_type, {384, 768}});
        auto y  = m2.add_parameter("y", {migraphx::shape::int32_type, {768}});
        auto yb = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {384, 768}}}), y);
        auto sum  = m2.add_instruction(migraphx::make_op("add"), x, yb);
        auto sumb = m2.add_instruction(b, sum);
        m2.add_instruction(pass_op{}, sumb);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_inner_broadcast_different_dims_single_element)
{
    auto b = migraphx::make_op("multibroadcast", {{"out_lens", {2, 1, 4, 5}}});
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::int32_type, {1, 1, 1}});
        auto y   = m1.add_parameter("y", {migraphx::shape::int32_type, {1, 1, 1, 1}});
        auto xb  = m1.add_instruction(b, x);
        auto yb  = m1.add_instruction(b, y);
        auto sum = m1.add_instruction(migraphx::make_op("add"), xb, yb);
        m1.add_instruction(pass_op{}, sum);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", {migraphx::shape::int32_type, {1, 1, 1}});
        auto y  = m2.add_parameter("y", {migraphx::shape::int32_type, {1, 1, 1, 1}});
        auto xs = m2.add_instruction(migraphx::make_op("squeeze"), x);
        auto xb = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 1, 1}}}), xs);
        auto sum  = m2.add_instruction(migraphx::make_op("add"), xb, y);
        auto sumb = m2.add_instruction(b, sum);
        m2.add_instruction(pass_op{}, sumb);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_inner_broadcast_different_dims_single_element_no_squeeze)
{
    auto b = migraphx::make_op("multibroadcast", {{"out_lens", {2, 1, 4, 5}}});
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y   = m1.add_parameter("y", {migraphx::shape::int32_type, {1, 1, 1, 5}});
        auto z   = m1.add_parameter("z", {migraphx::shape::int32_type, {1, 1, 5}});
        auto xb  = m1.add_instruction(b, x);
        auto yb  = m1.add_instruction(b, y);
        auto zb  = m1.add_instruction(b, z);
        auto sum = m1.add_instruction(migraphx::make_op("where"), xb, yb, zb);
        m1.add_instruction(pass_op{}, sum);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = m2.add_parameter("y", {migraphx::shape::int32_type, {1, 1, 1, 5}});
        auto z    = m2.add_parameter("z", {migraphx::shape::int32_type, {1, 1, 5}});
        auto ys   = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 1, 2}}}), y);
        auto zs   = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 1}}}), z);
        auto xb   = m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {5}}}), x);
        auto sum  = m2.add_instruction(migraphx::make_op("where"), xb, ys, zs);
        auto sumb = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 3}, {"out_lens", {2, 1, 4, 5}}}), sum);
        m2.add_instruction(pass_op{}, sumb);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_inner_broadcast_different_dims_broadcasted)
{
    auto b = migraphx::make_op("multibroadcast", {{"out_lens", {2, 384, 768}}});
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::int32_type, {1, 768}});
        auto y = m1.add_parameter("y", {migraphx::shape::int32_type, {1, 768}});
        auto xb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {384, 768}}}), x);
        auto xbb = m1.add_instruction(b, xb);
        auto yb  = m1.add_instruction(b, y);
        auto sum = m1.add_instruction(migraphx::make_op("add"), xbb, yb);
        m1.add_instruction(pass_op{}, sum);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", {migraphx::shape::int32_type, {1, 768}});
        auto y    = m2.add_parameter("y", {migraphx::shape::int32_type, {1, 768}});
        auto sum  = m2.add_instruction(migraphx::make_op("add"), x, y);
        auto sumb = m2.add_instruction(b, sum);
        m2.add_instruction(pass_op{}, sumb);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_inner_broadcast_different_dims_broadcasted_scalar)
{
    auto b = migraphx::make_op("multibroadcast", {{"out_lens", {2, 384}}});
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y = m1.add_parameter("y", {migraphx::shape::int32_type, {1, 384}});
        auto xb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 1}}}), x);
        auto xbb = m1.add_instruction(b, xb);
        auto yb  = m1.add_instruction(b, y);
        auto sum = m1.add_instruction(migraphx::make_op("add"), xbb, yb);
        m1.add_instruction(pass_op{}, sum);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y = m2.add_parameter("y", {migraphx::shape::int32_type, {1, 384}});
        auto xb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 384}}}), x);
        auto sum  = m2.add_instruction(migraphx::make_op("add"), xb, y);
        auto sumb = m2.add_instruction(b, sum);
        m2.add_instruction(pass_op{}, sumb);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_inner_broadcast_different_broadcasts)
{
    auto b  = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 24, 112, 112}}});
    auto mb = migraphx::make_op("multibroadcast", {{"out_lens", {1, 24, 112, 112}}});
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::int32_type, {24}});
        auto y   = m1.add_parameter("y", {migraphx::shape::int32_type, {24, 1, 1}});
        auto xb  = m1.add_instruction(b, x);
        auto yb  = m1.add_instruction(mb, y);
        auto sum = m1.add_instruction(migraphx::make_op("add"), xb, yb);
        m1.add_instruction(pass_op{}, sum);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", {migraphx::shape::int32_type, {24}});
        auto y    = m2.add_parameter("y", {migraphx::shape::int32_type, {24, 1, 1}});
        auto ys   = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1, 2}}}), y);
        auto sum  = m2.add_instruction(migraphx::make_op("add"), x, ys);
        auto sumb = m2.add_instruction(b, sum);
        m2.add_instruction(pass_op{}, sumb);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_inner_broadcast_different_broadcasts_scalar_vector)
{
    auto b  = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 24, 112, 112}}});
    auto mb = migraphx::make_op("multibroadcast", {{"out_lens", {1, 24, 112, 112}}});
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y   = m1.add_parameter("y", {migraphx::shape::int32_type, {24}});
        auto xb  = m1.add_instruction(mb, x);
        auto yb  = m1.add_instruction(b, y);
        auto sum = m1.add_instruction(migraphx::make_op("add"), xb, yb);
        m1.add_instruction(pass_op{}, sum);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y   = m2.add_parameter("y", {migraphx::shape::int32_type, {24}});
        auto xb  = m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {24}}}), x);
        auto sum = m2.add_instruction(migraphx::make_op("add"), xb, y);
        auto sumb = m2.add_instruction(b, sum);
        m2.add_instruction(pass_op{}, sumb);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_inner_broadcast_different_broadcasts_diff_axis)
{
    auto b  = migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {1, 64, 112, 112}}});
    auto mb = migraphx::make_op("multibroadcast", {{"out_lens", {1, 64, 112, 112}}});
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::int32_type, {1, 64}});
        auto y   = m1.add_parameter("y", {migraphx::shape::int32_type, {64, 1, 1}});
        auto xb  = m1.add_instruction(b, x);
        auto yb  = m1.add_instruction(mb, y);
        auto sum = m1.add_instruction(migraphx::make_op("add"), xb, yb);
        m1.add_instruction(pass_op{}, sum);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", {migraphx::shape::int32_type, {1, 64}});
        auto y    = m2.add_parameter("y", {migraphx::shape::int32_type, {64, 1, 1}});
        auto xs   = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), x);
        auto ys   = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1, 2}}}), y);
        auto sum  = m2.add_instruction(migraphx::make_op("add"), xs, ys);
        auto sumb = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 64, 112, 112}}}), sum);
        m2.add_instruction(pass_op{}, sumb);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_inner_broadcast_different_broadcasts_diff_dims)
{
    auto b = migraphx::make_op("multibroadcast", {{"out_lens", {1, 5, 10}}});
    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", {migraphx::shape::int32_type, {256}});
        auto y  = m1.add_parameter("y", {migraphx::shape::int32_type, {1, 256, 31, 31}});
        auto xb = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {64, 256, 31, 31}}}), x);
        auto yb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {64, 256, 31, 31}}}), y);
        auto sum = m1.add_instruction(migraphx::make_op("add"), xb, yb);
        m1.add_instruction(pass_op{}, sum);
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", {migraphx::shape::int32_type, {256}});
        auto y  = m2.add_parameter("y", {migraphx::shape::int32_type, {1, 256, 31, 31}});
        auto xb = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {256, 31, 31}}}), x);
        auto ys   = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), y);
        auto sum  = m2.add_instruction(migraphx::make_op("add"), xb, ys);
        auto sumb = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {64, 256, 31, 31}}}), sum);
        m2.add_instruction(pass_op{}, sumb);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_inner_broadcast_no_common_axis)
{
    auto b = migraphx::make_op("multibroadcast", {{"out_lens", {1, 5, 10}}});
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::int32_type, {5, 10}});
        auto y   = m1.add_parameter("y", {migraphx::shape::int32_type, {1, 5, 1}});
        auto xb  = m1.add_instruction(b, x);
        auto yb  = m1.add_instruction(b, y);
        auto sum = m1.add_instruction(migraphx::make_op("add"), xb, yb);
        m1.add_instruction(pass_op{}, sum);
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_add_conv1)
{
    migraphx::module m;
    auto x = m.add_parameter("x", {migraphx::shape::float_type, {1, 128, 28, 28}});
    auto w =
        m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 3, 3}}));
    auto y = m.add_parameter("y", {migraphx::shape::float_type, {1, 128, 28, 28}});
    auto v =
        m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 3, 3}}));
    auto conv1 = m.add_instruction(migraphx::make_op("convolution"), x, w);
    auto conv2 = m.add_instruction(migraphx::make_op("convolution"), y, v);
    auto sum   = m.add_instruction(migraphx::make_op("add"), conv1, conv2);
    m.add_instruction(pass_op{}, sum);
    auto s = m.get_output_shapes().back();
    run_pass(m);
    EXPECT(s == m.get_output_shapes().back());
    EXPECT(std::count_if(
               m.begin(), m.end(), [](auto&& ins) { return ins.name() == "convolution"; }) == 1);
}

TEST_CASE(simplify_add_conv_no_fusion_7x7_diff_strides)
{
    migraphx::module m;
    auto x = m.add_parameter("x", {migraphx::shape::float_type, {1, 128, 14, 14}});
    auto w =
        m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 7, 7}}));
    auto y = m.add_parameter("y", {migraphx::shape::float_type, {1, 128, 28, 28}});
    auto v =
        m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 7, 7}}));
    auto conv1 = m.add_instruction(migraphx::make_op("convolution"), x, w);
    auto conv2 = m.add_instruction(
        migraphx::make_op("convolution", {{"padding", {0, 0}}, {"stride", {3, 3}}}), y, v);
    auto sum = m.add_instruction(migraphx::make_op("add"), conv1, conv2);
    m.add_instruction(pass_op{}, sum);
    auto s = m.get_output_shapes().back();
    run_pass(m);
    EXPECT(s == m.get_output_shapes().back());
    // No fusion
    EXPECT(std::count_if(
               m.begin(), m.end(), [](auto&& ins) { return ins.name() == "convolution"; }) == 2);
}

TEST_CASE(simplify_add_conv_1x1_diff_strides1)
{
    migraphx::module m;
    auto x = m.add_parameter("x", {migraphx::shape::float_type, {1, 128, 14, 14}});
    auto w =
        m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto y = m.add_parameter("y", {migraphx::shape::float_type, {1, 128, 28, 28}});
    auto v =
        m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto conv1 = m.add_instruction(migraphx::make_op("convolution"), x, w);
    auto conv2 = m.add_instruction(
        migraphx::make_op("convolution", {{"padding", {0, 0}}, {"stride", {2, 2}}}), y, v);
    auto sum = m.add_instruction(migraphx::make_op("add"), conv1, conv2);
    m.add_instruction(pass_op{}, sum);
    auto s = m.get_output_shapes().back();
    run_pass(m);
    EXPECT(s == m.get_output_shapes().back());
    EXPECT(std::count_if(
               m.begin(), m.end(), [](auto&& ins) { return ins.name() == "convolution"; }) == 1);
}

TEST_CASE(simplify_add_conv_1x1_diff_strides2)
{
    migraphx::module m;
    auto x = m.add_parameter("x", {migraphx::shape::float_type, {1, 128, 28, 28}});
    auto w =
        m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto y = m.add_parameter("y", {migraphx::shape::float_type, {1, 128, 14, 14}});
    auto v =
        m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto conv1 = m.add_instruction(
        migraphx::make_op("convolution", {{"padding", {0, 0}}, {"stride", {2, 2}}}), x, w);
    auto conv2 = m.add_instruction(migraphx::make_op("convolution"), y, v);
    auto sum   = m.add_instruction(migraphx::make_op("add"), conv1, conv2);
    m.add_instruction(pass_op{}, sum);
    auto s = m.get_output_shapes().back();
    run_pass(m);
    EXPECT(s == m.get_output_shapes().back());
    EXPECT(std::count_if(
               m.begin(), m.end(), [](auto&& ins) { return ins.name() == "convolution"; }) == 1);
}

TEST_CASE(simplify_add_conv_1x1_diff_strides_odd)
{
    migraphx::module m;
    auto x = m.add_parameter("x", {migraphx::shape::float_type, {1, 54, 83, 83}});
    auto w =
        m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {54, 54, 1, 1}}));
    auto y = m.add_parameter("y", {migraphx::shape::float_type, {1, 54, 165, 165}});
    auto v =
        m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {54, 54, 1, 1}}));
    auto conv1 = m.add_instruction(migraphx::make_op("convolution"), x, w);
    auto conv2 = m.add_instruction(
        migraphx::make_op("convolution", {{"padding", {0, 0}}, {"stride", {2, 2}}}), y, v);
    auto sum = m.add_instruction(migraphx::make_op("add"), conv1, conv2);
    m.add_instruction(pass_op{}, sum);
    auto s = m.get_output_shapes().back();
    run_pass(m);
    EXPECT(s == m.get_output_shapes().back());
    EXPECT(std::count_if(
               m.begin(), m.end(), [](auto&& ins) { return ins.name() == "convolution"; }) == 1);
}

TEST_CASE(simplify_add_conv_no_fusion_asymetrical_strides1)
{
    migraphx::module m;
    auto x = m.add_parameter("x", {migraphx::shape::float_type, {1, 128, 28, 14}});
    auto w =
        m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto y = m.add_parameter("y", {migraphx::shape::float_type, {1, 128, 14, 14}});
    auto v =
        m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto conv1 = m.add_instruction(
        migraphx::make_op("convolution", {{"padding", {0, 0}}, {"stride", {2, 1}}}), x, w);
    auto conv2 = m.add_instruction(migraphx::make_op("convolution"), y, v);
    auto sum   = m.add_instruction(migraphx::make_op("add"), conv1, conv2);
    m.add_instruction(pass_op{}, sum);
    auto s = m.get_output_shapes().back();
    run_pass(m);
    EXPECT(s == m.get_output_shapes().back());
    // No fusion
    EXPECT(std::count_if(
               m.begin(), m.end(), [](auto&& ins) { return ins.name() == "convolution"; }) == 2);
}

TEST_CASE(simplify_add_conv_no_fusion_asymetrical_strides2)
{
    migraphx::module m;
    auto x = m.add_parameter("x", {migraphx::shape::float_type, {1, 128, 14, 14}});
    auto w =
        m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto y = m.add_parameter("y", {migraphx::shape::float_type, {1, 128, 28, 14}});
    auto v =
        m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto conv1 = m.add_instruction(migraphx::make_op("convolution"), x, w);
    auto conv2 = m.add_instruction(
        migraphx::make_op("convolution", {{"padding", {0, 0}}, {"stride", {2, 1}}}), y, v);
    auto sum = m.add_instruction(migraphx::make_op("add"), conv1, conv2);
    m.add_instruction(pass_op{}, sum);
    auto s = m.get_output_shapes().back();
    run_pass(m);
    EXPECT(s == m.get_output_shapes().back());
    // No fusion
    EXPECT(std::count_if(
               m.begin(), m.end(), [](auto&& ins) { return ins.name() == "convolution"; }) == 2);
}

TEST_CASE(simplify_concat_unpack_int4)
{
    auto s = migraphx::shape{migraphx::shape::int8_type, {11008, 2048}};
    migraphx::module m1;
    {
        auto x       = m1.add_parameter("x", s);
        auto y       = m1.add_parameter("y", s);
        auto unpack1 = m1.add_instruction(migraphx::make_op("unpack_int4"), x);
        auto unpack2 = m1.add_instruction(migraphx::make_op("unpack_int4"), y);
        auto concat =
            m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), unpack1, unpack2);
        m1.add_return({concat});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto y      = m2.add_parameter("y", s);
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y);
        auto unpack = m2.add_instruction(migraphx::make_op("unpack_int4"), concat);
        m2.add_return({unpack});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_concat_add_relu)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {1}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto one    = m1.add_literal({s, {1}});
        auto two    = m1.add_literal({s, {2}});
        auto sum1   = m1.add_instruction(migraphx::make_op("add"), x, one);
        auto relu1  = m1.add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2   = m1.add_instruction(migraphx::make_op("add"), y, two);
        auto relu2  = m1.add_instruction(migraphx::make_op("relu"), sum2);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), relu1, relu2);
        m1.add_instruction(pass_op{}, concat);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x       = m2.add_parameter("x", s);
        auto y       = m2.add_parameter("y", s);
        auto one     = m2.add_literal({s, {1}});
        auto two     = m2.add_literal({s, {2}});
        auto concat1 = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y);
        auto concat2 = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), one, two);
        auto sum     = m2.add_instruction(migraphx::make_op("add"), concat1, concat2);
        auto relu    = m2.add_instruction(migraphx::make_op("relu"), sum);
        m2.add_instruction(pass_op{}, relu);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_concat_add_relu_partial)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {1}};
    migraphx::module m1;
    {
        auto x     = m1.add_parameter("x", s);
        auto y     = m1.add_parameter("y", s);
        auto one   = m1.add_literal({s, {1}});
        auto two   = m1.add_literal({s, {2}});
        auto sum1  = m1.add_instruction(migraphx::make_op("add"), x, one);
        auto relu1 = m1.add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2  = m1.add_instruction(migraphx::make_op("add"), y, two);
        auto relu2 = m1.add_instruction(migraphx::make_op("relu"), sum2);
        auto sum3  = m1.add_instruction(migraphx::make_op("add"), x, y);
        auto concat =
            m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), sum3, relu1, relu2);
        m1.add_instruction(pass_op{}, concat);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x       = m2.add_parameter("x", s);
        auto y       = m2.add_parameter("y", s);
        auto one     = m2.add_literal({s, {1}});
        auto two     = m2.add_literal({s, {2}});
        auto concat1 = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y);
        auto concat2 = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), one, two);
        auto sum1    = m2.add_instruction(migraphx::make_op("add"), concat1, concat2);
        auto relu    = m2.add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2    = m2.add_instruction(migraphx::make_op("add"), x, y);
        auto concat  = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), sum2, relu);
        m2.add_instruction(pass_op{}, concat);
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_concat_add_relu_partial_broadcast)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {2, 1, 4, 5}};
    migraphx::module m1;
    {
        auto b    = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 1, 4, 5}}});
        auto x    = m1.add_parameter("x", s);
        auto y    = m1.add_parameter("y", s);
        auto one  = m1.add_literal(1);
        auto oneb = m1.add_instruction(b, one);
        auto two  = m1.add_literal(2);
        auto twob = m1.add_instruction(b, two);
        auto sum  = m1.add_instruction(migraphx::make_op("add"), x, y);
        auto concat =
            m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), sum, oneb, twob);
        m1.add_instruction(pass_op{}, concat);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto b       = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 2, 4, 5}}});
        auto x       = m2.add_parameter("x", s);
        auto y       = m2.add_parameter("y", s);
        auto one     = m2.add_literal(1);
        auto two     = m2.add_literal(2);
        auto concat1 = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), one, two);
        auto concatb = m2.add_instruction(b, concat1);
        auto sum     = m2.add_instruction(migraphx::make_op("add"), x, y);
        auto concat2 = m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), sum, concatb);
        m2.add_instruction(pass_op{}, concat2);
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_concat_add_relu_broadcast_different_axis)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {2, 1, 4, 5}};
    migraphx::module m1;
    {
        auto b      = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 1, 4, 5}}});
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto one    = m1.add_literal(1);
        auto oneb   = m1.add_instruction(b, one);
        auto two    = m1.add_literal(2);
        auto twob   = m1.add_instruction(b, two);
        auto sum1   = m1.add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1  = m1.add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2   = m1.add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2  = m1.add_instruction(migraphx::make_op("relu"), sum2);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), relu1, relu2);
        m1.add_instruction(pass_op{}, concat);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto b        = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 2, 4, 5}}});
        auto x        = m2.add_parameter("x", s);
        auto y        = m2.add_parameter("y", s);
        auto one      = m2.add_literal(1);
        auto two      = m2.add_literal(2);
        auto concat1  = m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
        auto concat2  = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), one, two);
        auto concat2b = m2.add_instruction(b, concat2);
        auto sum      = m2.add_instruction(migraphx::make_op("add"), concat1, concat2b);
        auto relu     = m2.add_instruction(migraphx::make_op("relu"), sum);
        m2.add_instruction(pass_op{}, relu);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_concat_add_relu_broadcast_same_axis)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {2, 1, 4, 5}};
    migraphx::module m1;
    {
        auto b      = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 1, 4, 5}}});
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto one    = m1.add_literal(1);
        auto oneb   = m1.add_instruction(b, one);
        auto two    = m1.add_literal(2);
        auto twob   = m1.add_instruction(b, two);
        auto sum1   = m1.add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1  = m1.add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2   = m1.add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2  = m1.add_instruction(migraphx::make_op("relu"), sum2);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), relu1, relu2);
        m1.add_instruction(pass_op{}, concat);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto b       = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 1, 4, 5}}});
        auto x       = m2.add_parameter("x", s);
        auto y       = m2.add_parameter("y", s);
        auto one     = m2.add_literal(1);
        auto oneb    = m2.add_instruction(b, one);
        auto two     = m2.add_literal(2);
        auto twob    = m2.add_instruction(b, two);
        auto concat1 = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y);
        auto concat2 = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), oneb, twob);
        auto sum     = m2.add_instruction(migraphx::make_op("add"), concat1, concat2);
        auto relu    = m2.add_instruction(migraphx::make_op("relu"), sum);
        m2.add_instruction(pass_op{}, relu);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_concat_clip)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {1}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto min    = m1.add_literal({s, {0}});
        auto max    = m1.add_literal({s, {10}});
        auto clip1  = m1.add_instruction(migraphx::make_op("clip"), x, min, max);
        auto clip2  = m1.add_instruction(migraphx::make_op("clip"), y, min, max);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), clip1, clip2);
        m1.add_instruction(pass_op{}, concat);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x       = m2.add_parameter("x", s);
        auto y       = m2.add_parameter("y", s);
        auto min     = m2.add_literal({s, {0}});
        auto max     = m2.add_literal({s, {10}});
        auto concat1 = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y);
        auto concat2 = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), min, min);
        auto concat3 = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), max, max);
        auto clip    = m2.add_instruction(migraphx::make_op("clip"), concat1, concat2, concat3);
        m2.add_instruction(pass_op{}, clip);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(concat_convert_fusion)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {64}};
    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", s);
        auto y  = m1.add_parameter("y", s);
        auto xh = m1.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::half_type)}}),
            x);
        auto yh = m1.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::half_type)}}),
            y);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), xh, yh);
        m1.add_instruction(pass_op{}, concat);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x       = m2.add_parameter("x", s);
        auto y       = m2.add_parameter("y", s);
        auto concat  = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y);
        auto concath = m2.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::half_type)}}),
            concat);
        m2.add_instruction(pass_op{}, concath);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_div_const)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::float_type, {1}});
        auto two = m1.add_literal(2.0f);
        m1.add_instruction(migraphx::make_op("div"), x, two);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x     = m2.add_parameter("x", {migraphx::shape::float_type, {1}});
        auto two   = m2.add_literal(2.0f);
        auto recip = m2.insert_instruction(std::next(two), migraphx::make_op("recip"), two);
        m2.add_instruction(migraphx::make_op("mul"), x, recip);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_div_integral_const)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto two = m1.add_literal(2);
        m1.add_instruction(migraphx::make_op("div"), x, two);
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_unit_mult_const)
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto unit = m1.add_literal(1);
        m1.add_instruction(migraphx::make_op("mul"), x, unit);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        m2.add_instruction(migraphx::make_op("identity"), x);
    }

    EXPECT(m1 == m2);
}

TEST_CASE(simplify_unit_mult_const2)
{
    migraphx::module m1;
    {
        auto unit = m1.add_literal(1);
        auto x    = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        m1.add_instruction(migraphx::make_op("mul"), unit, x);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        m2.add_instruction(migraphx::make_op("identity"), x);
    }

    EXPECT(m1 == m2);
}

TEST_CASE(simplify_unit_mult_const_vec)
{
    migraphx::shape unit_shape{migraphx::shape::int32_type, {2}};
    migraphx::shape x_shape{migraphx::shape::int32_type, {1, 2, 3, 3}};
    migraphx::module m1;
    {
        auto unit  = m1.add_literal({unit_shape, {1, 1}});
        auto x     = m1.add_parameter("x", x_shape);
        auto unitb = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 2, 3, 3}}}), unit);
        m1.add_instruction(migraphx::make_op("mul"), x, unitb);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", x_shape);
        m2.add_instruction(migraphx::make_op("identity"), x);
    }

    EXPECT(m1 == m2);
}

TEST_CASE(simplify_unit_mult_const_vec2)
{
    migraphx::shape unit_shape{migraphx::shape::int32_type, {2}};
    migraphx::shape x_shape{migraphx::shape::int32_type, {1, 2, 3, 3}};
    migraphx::module m1;
    {
        auto unit  = m1.add_literal({unit_shape, {1, 1}});
        auto x     = m1.add_parameter("x", x_shape);
        auto unitb = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 2, 3, 3}}}), unit);
        m1.add_instruction(migraphx::make_op("mul"), unitb, x);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", x_shape);
        m2.add_instruction(migraphx::make_op("identity"), x);
    }

    EXPECT(m1 == m2);
}

TEST_CASE(simplify_unit_div_const)
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto unit = m1.add_literal(1);
        auto div  = m1.add_instruction(migraphx::make_op("div"), x, unit);
        m1.add_return({div});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        m2.add_return({x});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(simplify_unit_div_const_vec)
{
    migraphx::shape unit_shape{migraphx::shape::int32_type, {2}};
    migraphx::shape x_shape{migraphx::shape::int32_type, {1, 2, 3, 3}};
    migraphx::module m1;
    {
        auto unit  = m1.add_literal({unit_shape, {1, 1}});
        auto x     = m1.add_parameter("x", x_shape);
        auto unitb = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 2, 3, 3}}}), unit);
        m1.add_instruction(migraphx::make_op("div"), x, unitb);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", x_shape);
        m2.add_instruction(migraphx::make_op("identity"), x);
    }

    EXPECT(m1 == m2);
}

TEST_CASE(simplify_neg_unit_mult_const)
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::int32_type, {1, 6}});
        auto unit = m1.add_literal(
            migraphx::literal{{migraphx::shape::int32_type, {1, 6}}, std::vector<int>(6, -1)});
        m1.add_instruction(migraphx::make_op("mul"), x, unit);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", {migraphx::shape::int32_type, {1, 6}});
        auto x2 = m2.add_instruction(migraphx::make_op("neg"), x);
        m2.add_instruction(migraphx::make_op("identity"), x2);
    }

    EXPECT((m1 == m2));
}

TEST_CASE(simplify_neg_unit_mult_const2)
{
    migraphx::module m1;
    {
        auto unit = m1.add_literal(-1);
        auto x    = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        m1.add_instruction(migraphx::make_op("mul"), unit, x);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto x2 = m2.add_instruction(migraphx::make_op("neg"), x);
        m2.add_instruction(migraphx::make_op("identity"), x2);
    }

    EXPECT((m1 == m2));
}

TEST_CASE(simplify_neg_unit_mult_const_add)
{
    migraphx::module m1;
    {
        auto unit = m1.add_literal(-1);
        auto x    = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto x2   = m1.add_instruction(migraphx::make_op("mul"), unit, x);
        m1.add_instruction(migraphx::make_op("add"), x2, x2);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto x2 = m2.add_instruction(migraphx::make_op("neg"), x);
        m2.add_instruction(migraphx::make_op("add"), x2, x2);
    }

    EXPECT((m1 == m2));
}

TEST_CASE(simplify_neg_unit_mul_const_vec)
{
    migraphx::shape unit_shape{migraphx::shape::int32_type, {2}};
    migraphx::shape x_shape{migraphx::shape::int32_type, {1, 2, 3, 3}};
    migraphx::module m1;
    {
        auto unit  = m1.add_literal({unit_shape, {-1, -1}});
        auto x     = m1.add_parameter("x", x_shape);
        auto unitb = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 2, 3, 3}}}), unit);
        m1.add_instruction(migraphx::make_op("mul"), x, unitb);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", x_shape);
        auto x2 = m2.add_instruction(migraphx::make_op("neg"), x);
        m2.add_instruction(migraphx::make_op("identity"), x2);
    }

    EXPECT(m1 == m2);
}

TEST_CASE(simplify_neg_unit_mul_const_vec2)
{
    migraphx::shape zero_shape{migraphx::shape::int32_type, {2}};
    migraphx::shape x_shape{migraphx::shape::int32_type, {1, 2, 3, 3}};
    migraphx::module m1;
    {
        auto unit  = m1.add_literal({zero_shape, {-1, -1}});
        auto x     = m1.add_parameter("x", x_shape);
        auto unitb = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 2, 3, 3}}}), unit);
        m1.add_instruction(migraphx::make_op("mul"), unitb, x);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", x_shape);
        auto x2 = m2.add_instruction(migraphx::make_op("neg"), x);
        m2.add_instruction(migraphx::make_op("identity"), x2);
    }

    EXPECT(m1 == m2);
}

TEST_CASE(simplify_neg_unit_div_const)
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto unit = m1.add_literal(-1);
        m1.add_instruction(migraphx::make_op("div"), x, unit);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto x2 = m2.add_instruction(migraphx::make_op("neg"), x);
        m2.add_instruction(migraphx::make_op("identity"), x2);
    }

    EXPECT(m1 == m2);
}

TEST_CASE(simplify_neg_unit_div_const_vec)
{
    migraphx::shape unit_shape{migraphx::shape::int32_type, {2}};
    migraphx::shape x_shape{migraphx::shape::int32_type, {1, 2, 3, 3}};
    migraphx::module m1;
    {
        auto unit  = m1.add_literal({unit_shape, {-1, -1}});
        auto x     = m1.add_parameter("x", x_shape);
        auto unitb = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 2, 3, 3}}}), unit);
        m1.add_instruction(migraphx::make_op("div"), x, unitb);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", x_shape);
        auto x2 = m2.add_instruction(migraphx::make_op("neg"), x);
        m2.add_instruction(migraphx::make_op("identity"), x2);
    }

    EXPECT(m1 == m2);
}

TEST_CASE(simplify_sub_zero_const)
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto zero = m1.add_literal(0);
        m1.add_instruction(migraphx::make_op("sub"), x, zero);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        m2.add_instruction(migraphx::make_op("identity"), x);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_sub_zero_const_vec)
{
    migraphx::shape zero_shape{migraphx::shape::int32_type, {2}};
    migraphx::shape x_shape{migraphx::shape::int32_type, {1, 2, 3, 3}};
    migraphx::module m1;
    {
        auto zero  = m1.add_literal({zero_shape, {0, 0}});
        auto x     = m1.add_parameter("x", x_shape);
        auto zerob = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 2, 3, 3}}}), zero);
        m1.add_instruction(migraphx::make_op("sub"), x, zerob);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", x_shape);
        m2.add_instruction(migraphx::make_op("identity"), x);
    }

    EXPECT(m1 == m2);
}

TEST_CASE(simplify_sub_neg_zero_const)
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto zero = m1.add_literal(0);
        m1.add_instruction(migraphx::make_op("sub"), zero, x);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto x2 = m2.add_instruction(migraphx::make_op("neg"), x);
        m2.add_instruction(migraphx::make_op("identity"), x2);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_sub_neg_zero_const_vec)
{
    migraphx::shape zero_shape{migraphx::shape::int32_type, {2}};
    migraphx::shape x_shape{migraphx::shape::int32_type, {1, 2, 3, 3}};
    migraphx::module m1;
    {
        auto zero  = m1.add_literal({zero_shape, {0, 0}});
        auto x     = m1.add_parameter("x", x_shape);
        auto zerob = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 2, 3, 3}}}), zero);
        m1.add_instruction(migraphx::make_op("sub"), zerob, x);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", x_shape);
        auto x2 = m2.add_instruction(migraphx::make_op("neg"), x);
        m2.add_instruction(migraphx::make_op("identity"), x2);
    }

    EXPECT(m1 == m2);
}

TEST_CASE(simplify_sub_const)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto two = m1.add_literal(2);
        m1.add_instruction(migraphx::make_op("sub"), x, two);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto two = m2.add_literal(2);
        auto neg = m2.insert_instruction(std::next(two), migraphx::make_op("neg"), two);
        m2.add_instruction(migraphx::make_op("add"), x, neg);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_zero_mult_const)
{
    migraphx::module m1;
    {
        auto x       = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto zero    = m1.add_literal(0);
        auto mul_ins = m1.add_instruction(migraphx::make_op("mul"), x, zero);
        m1.add_return({mul_ins});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto zero = m2.add_literal(0);
        m2.add_return({zero});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(simplify_zero_mult_const2)
{
    migraphx::module m1;
    {
        auto x       = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto zero    = m1.add_literal(0);
        auto mul_ins = m1.add_instruction(migraphx::make_op("mul"), zero, x);
        m1.add_return({mul_ins});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto zero = m2.add_literal(0);
        m2.add_return({zero});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(simplify_zero_mul_const_vec)
{
    migraphx::shape zero_shape{migraphx::shape::int32_type, {2}};
    migraphx::shape x_shape{migraphx::shape::int32_type, {1, 2, 3, 3}};
    migraphx::module m1;
    {
        auto zero  = m1.add_literal({zero_shape, {0, 0}});
        auto x     = m1.add_parameter("x", x_shape);
        auto zerob = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 2, 3, 3}}}), zero);
        auto mul_ins = m1.add_instruction(migraphx::make_op("mul"), x, zerob);
        m1.add_return({mul_ins});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto zero = m2.add_literal({zero_shape, {0, 0}});
        m2.add_parameter("x", x_shape);
        auto zerob = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 2, 3, 3}}}), zero);
        m2.add_return({zerob});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(simplify_zero_mul_const_vec2)
{
    migraphx::shape zero_shape{migraphx::shape::int32_type, {2}};
    migraphx::shape x_shape{migraphx::shape::int32_type, {1, 2, 3, 3}};
    migraphx::module m1;
    {
        auto zero  = m1.add_literal({zero_shape, {0, 0}});
        auto x     = m1.add_parameter("x", x_shape);
        auto zerob = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 2, 3, 3}}}), zero);
        auto mul_ins = m1.add_instruction(migraphx::make_op("mul"), zerob, x);
        m1.add_return({mul_ins});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto zero = m2.add_literal({zero_shape, {0, 0}});
        m2.add_parameter("x", x_shape);
        auto zerob = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 2, 3, 3}}}), zero);
        m2.add_return({zerob});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(simplify_zero_div_const)
{
    migraphx::module m1;
    {
        auto zero    = m1.add_literal(0);
        auto x       = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto div_ins = m1.add_instruction(migraphx::make_op("div"), zero, x);
        m1.add_return({div_ins});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto zero = m2.add_literal(0);
        m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        m2.add_return({zero});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(simplify_zero_div_const_vec)
{
    migraphx::shape zero_shape{migraphx::shape::int32_type, {2}};
    migraphx::shape x_shape{migraphx::shape::int32_type, {1, 2, 3, 3}};
    migraphx::module m1;
    {
        auto x     = m1.add_parameter("x", x_shape);
        auto zero  = m1.add_literal({zero_shape, {0, 0}});
        auto zerob = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 2, 3, 3}}}), zero);
        auto div_ins = m1.add_instruction(migraphx::make_op("div"), zerob, x);
        m1.add_return({div_ins});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        m2.add_parameter("x", x_shape);
        auto zero  = m2.add_literal({zero_shape, {0, 0}});
        auto zerob = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 2, 3, 3}}}), zero);
        m2.add_return({zerob});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(simplify_rsqrt)
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto sqrt = m1.add_instruction(migraphx::make_op("sqrt"), x);
        m1.add_instruction(migraphx::make_op("recip"), sqrt);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        m2.add_instruction(migraphx::make_op("rsqrt"), x);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_rsqrt_multi_use)
{
    migraphx::module m1;
    {
        auto x     = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto sqrt  = m1.add_instruction(migraphx::make_op("sqrt"), x);
        auto add   = m1.add_instruction(migraphx::make_op("add"), sqrt, sqrt);
        auto rsqrt = m1.add_instruction(migraphx::make_op("recip"), sqrt);
        m1.add_instruction(migraphx::make_op("add"), rsqrt, add);
    }
    migraphx::module m2{m1};

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_slice_concat)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {256}};

    migraphx::module m1;
    {
        auto x       = m1.add_parameter("x", s);
        auto y       = m1.add_parameter("y", s);
        auto xslice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {128}}}), x);
        auto xslice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {128}}, {"ends", {256}}}), x);
        auto yslice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {128}}}), y);
        auto yslice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {128}}, {"ends", {256}}}), y);
        auto concat = m1.add_instruction(
            migraphx::make_op("concat", {{"axis", 0}}), xslice1, xslice2, yslice1, yslice2);
        m1.add_instruction(pass_op{}, concat);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto y      = m2.add_parameter("y", s);
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y);
        m2.add_instruction(pass_op{}, concat);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(simplify_slice_concat_non_uniform)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {256}};

    migraphx::module m1;
    {
        auto x       = m1.add_parameter("x", s);
        auto y       = m1.add_parameter("y", s);
        auto xslice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {64}}}), x);
        auto xslice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {64}}, {"ends", {192}}}), x);
        auto xslice3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {192}}, {"ends", {256}}}), x);
        auto yslice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {64}}}), y);
        auto yslice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {64}}, {"ends", {192}}}), y);
        auto yslice3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {192}}, {"ends", {256}}}), y);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                         xslice1,
                                         xslice2,
                                         xslice3,
                                         yslice1,
                                         yslice2,
                                         yslice3);
        m1.add_instruction(pass_op{}, concat);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto y      = m2.add_parameter("y", s);
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y);
        m2.add_instruction(pass_op{}, concat);
    }

    EXPECT(m1 == m2);
}

TEST_CASE(simplify_slice_concat_flipped)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {256}};

    migraphx::module m1;
    {
        auto x       = m1.add_parameter("x", s);
        auto y       = m1.add_parameter("y", s);
        auto xslice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {64}}}), x);
        auto xslice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {192}}, {"ends", {256}}}), x);
        auto xslice3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {64}}, {"ends", {192}}}), x);
        auto yslice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {64}}}), y);
        auto yslice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {192}}, {"ends", {256}}}), y);
        auto yslice3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {64}}, {"ends", {192}}}), y);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                         xslice1,
                                         xslice2,
                                         xslice3,
                                         yslice1,
                                         yslice2,
                                         yslice3);
        m1.add_instruction(pass_op{}, concat);
    }
    migraphx::module m2 = m1;
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(simplify_slice_concat_interleaved_non_slice)
{
    // Matched by matcher find_split_concat, but no substitution because the "slice"
    // instructions in the input list have a different instruction mixed in
    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {4, 3, 3, 3}};
        auto x      = m1.add_parameter("x", s);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {1}}}), x);
        auto relu   = m1.add_instruction(migraphx::make_op("relu"), x);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {3}}}), x);
        auto concat =
            m1.add_instruction(migraphx::make_op("concat", {{"axis", 2}}), slice1, relu, slice2);
        m1.add_instruction(pass_op{}, concat);
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_split_add_relu)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4}};
    migraphx::module m1;
    {
        auto b     = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {3, 1, 4}}});
        auto input = m1.add_parameter("input", s);
        auto x     = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto one   = m1.add_literal(1);
        auto oneb  = m1.add_instruction(b, one);
        auto two   = m1.add_literal(2);
        auto twob  = m1.add_instruction(b, two);
        auto sum1  = m1.add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1 = m1.add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2  = m1.add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2 = m1.add_instruction(migraphx::make_op("relu"), sum2);
        auto add   = m1.add_instruction(migraphx::make_op("add"), relu1, relu2);
        m1.add_instruction(pass_op{}, add);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto b       = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {3, 2, 4}}});
        auto input   = m2.add_parameter("input", s);
        auto one     = m2.add_literal(1);
        auto two     = m2.add_literal(2);
        auto concat  = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), one, two);
        auto concatb = m2.add_instruction(b, concat);
        auto sum     = m2.add_instruction(migraphx::make_op("add"), input, concatb);
        auto relu    = m2.add_instruction(migraphx::make_op("relu"), sum);
        auto x       = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), relu);
        auto y = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), relu);
        auto add = m2.add_instruction(migraphx::make_op("add"), x, y);
        m2.add_instruction(pass_op{}, add);
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_split_add_flipped_input)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4}};
    migraphx::module m1;
    {
        auto b     = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {3, 1, 4}}});
        auto input = m1.add_parameter("input", s);
        auto x     = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto one   = m1.add_literal(1);
        auto oneb  = m1.add_instruction(b, one);
        auto two   = m1.add_literal(2);
        auto twob  = m1.add_instruction(b, two);
        auto sum1  = m1.add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1 = m1.add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2  = m1.add_instruction(migraphx::make_op("add"), twob, y);
        auto relu2 = m1.add_instruction(migraphx::make_op("relu"), sum2);
        auto add   = m1.add_instruction(migraphx::make_op("add"), relu1, relu2);
        m1.add_instruction(pass_op{}, add);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto b       = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {3, 2, 4}}});
        auto input   = m2.add_parameter("input", s);
        auto one     = m2.add_literal(1);
        auto two     = m2.add_literal(2);
        auto concat  = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), one, two);
        auto concatb = m2.add_instruction(b, concat);
        auto sum     = m2.add_instruction(migraphx::make_op("add"), input, concatb);
        auto relu    = m2.add_instruction(migraphx::make_op("relu"), sum);
        auto x       = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), relu);
        auto y = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), relu);
        auto add = m2.add_instruction(migraphx::make_op("add"), x, y);
        m2.add_instruction(pass_op{}, add);
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_split_non_comm_flipped_input)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4}};
    migraphx::module m1;
    {
        auto b     = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {3, 1, 4}}});
        auto input = m1.add_parameter("input", s);
        auto x     = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto one   = m1.add_literal(1);
        auto oneb  = m1.add_instruction(b, one);
        auto two   = m1.add_literal(2);
        auto twob  = m1.add_instruction(b, two);
        auto sum1  = m1.add_instruction(migraphx::make_op("sub"), x, oneb);
        auto relu1 = m1.add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2  = m1.add_instruction(migraphx::make_op("sub"), twob, y);
        auto relu2 = m1.add_instruction(migraphx::make_op("relu"), sum2);
        auto add   = m1.add_instruction(migraphx::make_op("sub"), relu1, relu2);
        m1.add_instruction(pass_op{}, add);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto b     = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {3, 1, 4}}});
        auto input = m2.add_parameter("input", s);
        auto x     = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto one     = m2.add_literal(1);
        auto neg_one = m2.add_instruction(migraphx::make_op("neg"), one);
        auto oneb    = m2.add_instruction(b, neg_one);
        auto two     = m2.add_literal(2);
        auto twob    = m2.add_instruction(b, two);
        auto sum1    = m2.add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1   = m2.add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2    = m2.add_instruction(migraphx::make_op("sub"), twob, y);
        auto relu2   = m2.add_instruction(migraphx::make_op("relu"), sum2);
        auto add     = m2.add_instruction(migraphx::make_op("sub"), relu1, relu2);
        m2.add_instruction(pass_op{}, add);
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_split_reduce0)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4}};
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", s);
        auto x     = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);

        auto one = m1.add_literal(1);
        auto two = m1.add_literal(2);

        auto arx   = m1.add_instruction(migraphx::make_op("contiguous"), x);
        auto ary   = m1.add_instruction(migraphx::make_op("contiguous"), y);
        auto rmax0 = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 1}}}), x);
        auto rmin0 = m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {0, 1}}}), x);
        auto rmax1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 1}}), arx, one);
        auto rmin1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 1}}), ary, two);
        auto rmax2 = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 1}}}), y);
        auto rmin2 = m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {0, 1}}}), y);
        m1.add_return({rmax0, rmin0, rmax1, rmin1, rmax2, rmin2});
    }

    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_split_reduce1)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4}};
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", s);
        auto x     = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);

        auto rmax0 = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 2}}}), x);
        auto rmin0 = m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {0, 2}}}), x);
        auto rmax2 = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 2}}}), y);
        auto rmin2 = m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {0, 2}}}), y);
        m1.add_return({rmax0, rmin0, rmax2, rmin2});
    }

    migraphx::module m2;
    {
        auto input = m2.add_parameter("input", s);

        auto rmn  = m2.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {0, 2}}}), input);
        auto slc0 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), rmn);
        auto rmx  = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 2}}}), input);
        auto slc1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), rmx);
        auto slc2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), rmn);
        auto slc3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), rmx);
        m2.add_return({slc3, slc2, slc1, slc0});
    }

    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_split_reduce2)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4}};
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", s);
        auto x     = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto rmax0 = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 2}}}), x);
        auto rmin0 = m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {0, 1}}}), x);
        auto rmax2 = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 2}}}), y);
        auto rmin2 = m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {0, 1}}}), y);
        m1.add_return({rmax0, rmin0, rmax2, rmin2});
    }

    migraphx::module m2;
    {
        auto input = m2.add_parameter("input", s);
        auto x     = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto rmn1 = m2.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {0, 1}}}), x);
        auto y    = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto rmn2 = m2.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {0, 1}}}), y);
        auto rms  = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 2}}}), input);
        auto slc0 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), rms);
        auto slc1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), rms);
        m2.add_return({slc1, rmn2, slc0, rmn1});
    }

    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_split_add_relu_reshape)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4}};
    migraphx::module m1;
    {
        auto b     = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {3, 1, 4}}});
        auto r     = migraphx::make_op("reshape", {{"dims", {3, 4}}});
        auto input = m1.add_parameter("input", s);
        auto x     = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto one      = m1.add_literal(1);
        auto oneb     = m1.add_instruction(b, one);
        auto two      = m1.add_literal(2);
        auto twob     = m1.add_instruction(b, two);
        auto sum1     = m1.add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1    = m1.add_instruction(migraphx::make_op("relu"), sum1);
        auto reshape1 = m1.add_instruction(r, relu1);
        auto sum2     = m1.add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2    = m1.add_instruction(migraphx::make_op("relu"), sum2);
        auto reshape2 = m1.add_instruction(r, relu2);
        auto add      = m1.add_instruction(migraphx::make_op("add"), reshape1, reshape2);
        m1.add_instruction(pass_op{}, add);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto b       = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {3, 2, 4}}});
        auto input   = m2.add_parameter("input", s);
        auto one     = m2.add_literal(1);
        auto two     = m2.add_literal(2);
        auto concat  = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), one, two);
        auto concatb = m2.add_instruction(b, concat);
        auto sum     = m2.add_instruction(migraphx::make_op("add"), input, concatb);
        auto relu    = m2.add_instruction(migraphx::make_op("relu"), sum);
        auto slc1    = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), relu);

        auto rsp1 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 4}}}), slc1);

        auto slc2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), relu);

        auto rsp2 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 4}}}), slc2);

        auto add = m2.add_instruction(migraphx::make_op("add"), rsp1, rsp2);
        m2.add_instruction(pass_op{}, add);
    }
    EXPECT(m1.sort() == m2.sort());
}

// Do not fuse with inter split group dependency
TEST_CASE(find_splits_inter_group_dependency)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4}};
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", s);
        auto x     = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto relu1 = m1.add_instruction(migraphx::make_op("relu"), x);
        auto sum1  = m1.add_instruction(migraphx::make_op("add"), relu1, x);
        auto relu2 = m1.add_instruction(migraphx::make_op("relu"), y);
        auto sum2  = m1.add_instruction(migraphx::make_op("add"), y, relu2);
        m1.add_return({sum1, sum2});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(find_splits_for_pointwise0)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {3, 2, 4}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto input = mm->add_parameter("input", s);
        auto x     = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto relu_x = add_pointwise(p1, "main:pointwise0", {x}, single_pointwise("relu"));
        auto relu_y = add_pointwise(p1, "main:pointwise1", {y}, single_pointwise("relu"));
        mm->add_return({relu_x, relu_y});
    }
    migraphx::run_passes(p1, {migraphx::simplify_algebra{}, migraphx::dead_code_elimination{}});

    migraphx::program p2;
    {
        auto* mm    = p2.get_main_module();
        auto input  = mm->add_parameter("input", s);
        auto relu   = add_pointwise(p2, "main:pointwise0", {input}, single_pointwise("relu"));
        auto relu_x = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), relu);
        auto relu_y = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), relu);
        mm->add_return({relu_x, relu_y});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(find_splits_for_pointwise1)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {3, 2, 4}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto input = mm->add_parameter("input", s);
        auto x     = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto sig_add_x =
            add_pointwise(p1, "main:pointwise0", {x}, [=](auto* pm, const auto& inputs) {
                auto sig = pm->add_instruction(migraphx::make_op("sigmoid"), inputs[0]);
                return pm->add_instruction(migraphx::make_op("add"), sig, inputs[0]);
            });
        auto sig_add_y =
            add_pointwise(p1, "main:pointwise1", {y}, [=](auto* pm, const auto& inputs) {
                auto sig = pm->add_instruction(migraphx::make_op("sigmoid"), inputs[0]);
                return pm->add_instruction(migraphx::make_op("add"), sig, inputs[0]);
            });
        mm->add_return({sig_add_x, sig_add_y});
    }
    migraphx::run_passes(p1, {migraphx::simplify_algebra{}, migraphx::dead_code_elimination{}});

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto input = mm->add_parameter("input", s);
        auto sig_add =
            add_pointwise(p2, "main:pointwise0", {input}, [=](auto* pm, const auto& inputs) {
                auto sig = pm->add_instruction(migraphx::make_op("sigmoid"), inputs[0]);
                return pm->add_instruction(migraphx::make_op("add"), sig, inputs[0]);
            });
        auto relu_x = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), sig_add);
        auto relu_y = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), sig_add);
        mm->add_return({relu_x, relu_y});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(find_splits_for_pointwise2)
{
    // Note: has inter-split dependency with sigmoid(x) + x so first simplify_algebra should be a
    // noop
    auto s = migraphx::shape{migraphx::shape::float_type, {3, 2, 4}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto input = mm->add_parameter("input", s);
        auto x     = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto sig_x = mm->add_instruction(migraphx::make_op("sigmoid"), x);
        auto add_x = mm->add_instruction(migraphx::make_op("add"), sig_x, x);
        auto sig_y = mm->add_instruction(migraphx::make_op("sigmoid"), y);
        auto add_y = mm->add_instruction(migraphx::make_op("add"), sig_y, y);
        mm->add_return({add_x, add_y});
    }
    migraphx::run_passes(p1,
                         {migraphx::simplify_algebra{},
                          migraphx::dead_code_elimination{},
                          migraphx::fuse_pointwise{},
                          migraphx::dead_code_elimination{},
                          migraphx::simplify_algebra{},
                          migraphx::dead_code_elimination{}});

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto input = mm->add_parameter("input", s);
        auto sig_add =
            add_pointwise(p2, "main:pointwise0", {input}, [=](auto* pm, const auto& inputs) {
                auto sig = pm->add_instruction(migraphx::make_op("sigmoid"), inputs[0]);
                return pm->add_instruction(migraphx::make_op("add"), sig, inputs[0]);
            });
        auto relu_x = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), sig_add);
        auto relu_y = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), sig_add);
        mm->add_return({relu_x, relu_y});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_slice_different_axis)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4, 2}};
    migraphx::module m1;
    {
        auto r     = migraphx::make_op("reshape", {{"dims", {3, 2, 4}}});
        auto input = m1.add_parameter("input", s);
        auto x     = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto one  = m1.add_literal(1);
        auto oneb = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {3, 1, 4, 2}}}), one);
        auto two  = m1.add_literal(2);
        auto twob = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 3}, {"out_lens", {3, 2, 4, 1}}}), two);
        auto sum1     = m1.add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1    = m1.add_instruction(migraphx::make_op("relu"), sum1);
        auto reshape1 = m1.add_instruction(r, relu1);
        auto sum2     = m1.add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2    = m1.add_instruction(migraphx::make_op("relu"), sum2);
        auto reshape2 = m1.add_instruction(r, relu2);
        auto add      = m1.add_instruction(migraphx::make_op("add"), reshape1, reshape2);
        m1.add_instruction(pass_op{}, add);
    }
    migraphx::module m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_slice_missing_begining_slice)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 3, 4}};
    migraphx::module m1;
    {
        auto b     = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {3, 1, 4}}});
        auto input = m1.add_parameter("input", s);
        auto x     = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {2}}, {"ends", {3}}}), input);
        auto y = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto one   = m1.add_literal(1);
        auto oneb  = m1.add_instruction(b, one);
        auto two   = m1.add_literal(2);
        auto twob  = m1.add_instruction(b, two);
        auto sum1  = m1.add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1 = m1.add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2  = m1.add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2 = m1.add_instruction(migraphx::make_op("relu"), sum2);
        auto add   = m1.add_instruction(migraphx::make_op("add"), relu1, relu2);
        m1.add_instruction(pass_op{}, add);
    }
    migraphx::module m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_slice_missing_middle_slice)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 3, 4}};
    migraphx::module m1;
    {
        auto b     = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {3, 1, 4}}});
        auto input = m1.add_parameter("input", s);
        auto x     = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {2}}, {"ends", {3}}}), input);
        auto y = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto one   = m1.add_literal(1);
        auto oneb  = m1.add_instruction(b, one);
        auto two   = m1.add_literal(2);
        auto twob  = m1.add_instruction(b, two);
        auto sum1  = m1.add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1 = m1.add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2  = m1.add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2 = m1.add_instruction(migraphx::make_op("relu"), sum2);
        auto add   = m1.add_instruction(migraphx::make_op("add"), relu1, relu2);
        m1.add_instruction(pass_op{}, add);
    }
    migraphx::module m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_slice_missing_end_slice)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 3, 4}};
    migraphx::module m1;
    {
        auto b     = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {3, 1, 4}}});
        auto input = m1.add_parameter("input", s);
        auto x     = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto one   = m1.add_literal(1);
        auto oneb  = m1.add_instruction(b, one);
        auto two   = m1.add_literal(2);
        auto twob  = m1.add_instruction(b, two);
        auto sum1  = m1.add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1 = m1.add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2  = m1.add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2 = m1.add_instruction(migraphx::make_op("relu"), sum2);
        auto add   = m1.add_instruction(migraphx::make_op("add"), relu1, relu2);
        m1.add_instruction(pass_op{}, add);
    }
    migraphx::module m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_split_add_relu_concat_same_axis)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4}};
    migraphx::module m1;
    {
        auto b     = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {3, 1, 4}}});
        auto input = m1.add_parameter("input", s);
        auto x     = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto one    = m1.add_literal(1);
        auto oneb   = m1.add_instruction(b, one);
        auto two    = m1.add_literal(2);
        auto twob   = m1.add_instruction(b, two);
        auto sum1   = m1.add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1  = m1.add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2   = m1.add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2  = m1.add_instruction(migraphx::make_op("relu"), sum2);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), relu1, relu2);
        m1.add_instruction(pass_op{}, concat);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto b       = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {3, 2, 4}}});
        auto input   = m2.add_parameter("input", s);
        auto one     = m2.add_literal(1);
        auto two     = m2.add_literal(2);
        auto concat  = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), one, two);
        auto concatb = m2.add_instruction(b, concat);
        auto sum     = m2.add_instruction(migraphx::make_op("add"), input, concatb);
        auto relu    = m2.add_instruction(migraphx::make_op("relu"), sum);
        m2.add_instruction(pass_op{}, relu);
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_split_add_relu_multi_axes)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4, 6}};
    migraphx::module m1;
    {
        auto b     = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {3, 1, 4, 3}}});
        auto input = m1.add_parameter("input", s);
        auto x     = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1, 3}}, {"starts", {0, 0}}, {"ends", {1, 3}}}),
            input);
        auto y = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1, 3}}, {"starts", {1, 3}}, {"ends", {2, 6}}}),
            input);
        auto one   = m1.add_literal(1);
        auto oneb  = m1.add_instruction(b, one);
        auto two   = m1.add_literal(2);
        auto twob  = m1.add_instruction(b, two);
        auto sum1  = m1.add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1 = m1.add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2  = m1.add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2 = m1.add_instruction(migraphx::make_op("relu"), sum2);
        auto add   = m1.add_instruction(migraphx::make_op("add"), relu1, relu2);
        m1.add_instruction(pass_op{}, add);
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_split_add_relu_used_multiple_split1)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4}};
    migraphx::module m1;
    {
        auto b     = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {3, 1, 4}}});
        auto input = m1.add_parameter("input", s);
        auto x     = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto one   = m1.add_literal(1);
        auto oneb  = m1.add_instruction(b, one);
        auto two   = m1.add_literal(2);
        auto twob  = m1.add_instruction(b, two);
        auto sum1  = m1.add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1 = m1.add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2  = m1.add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2 = m1.add_instruction(migraphx::make_op("relu"), sum2);
        auto add1  = m1.add_instruction(migraphx::make_op("add"), relu1, relu2);
        auto add2  = m1.add_instruction(migraphx::make_op("add"), x, add1);
        m1.add_instruction(pass_op{}, add2);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto b     = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {3, 2, 4}}});
        auto input = m2.add_parameter("input", s);
        auto slice = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto one     = m2.add_literal(1);
        auto two     = m2.add_literal(2);
        auto concat  = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), one, two);
        auto concatb = m2.add_instruction(b, concat);
        auto sum     = m2.add_instruction(migraphx::make_op("add"), input, concatb);
        auto relu    = m2.add_instruction(migraphx::make_op("relu"), sum);
        auto x       = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), relu);
        auto y = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), relu);
        auto add1 = m2.add_instruction(migraphx::make_op("add"), x, y);
        auto add2 = m2.add_instruction(migraphx::make_op("add"), slice, add1);
        m2.add_instruction(pass_op{}, add2);
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_split_add_relu_used_multiple_split2)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4}};
    migraphx::module m1;
    {
        auto b     = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {3, 1, 4}}});
        auto input = m1.add_parameter("input", s);
        auto x     = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto z     = m1.add_instruction(migraphx::make_op("relu"), x);
        auto one   = m1.add_literal(1);
        auto oneb  = m1.add_instruction(b, one);
        auto two   = m1.add_literal(2);
        auto twob  = m1.add_instruction(b, two);
        auto sum1  = m1.add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1 = m1.add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2  = m1.add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2 = m1.add_instruction(migraphx::make_op("relu"), sum2);
        auto add1  = m1.add_instruction(migraphx::make_op("add"), relu1, relu2);
        auto add2  = m1.add_instruction(migraphx::make_op("add"), z, add1);
        m1.add_instruction(pass_op{}, add2);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto b     = migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {3, 2, 4}}});
        auto input = m2.add_parameter("input", s);
        auto slice = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto z       = m2.add_instruction(migraphx::make_op("relu"), slice);
        auto one     = m2.add_literal(1);
        auto two     = m2.add_literal(2);
        auto concat  = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), one, two);
        auto concatb = m2.add_instruction(b, concat);
        auto sum     = m2.add_instruction(migraphx::make_op("add"), input, concatb);
        auto relu    = m2.add_instruction(migraphx::make_op("relu"), sum);
        auto x       = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), relu);
        auto y = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), relu);
        auto add1 = m2.add_instruction(migraphx::make_op("add"), x, y);
        auto add2 = m2.add_instruction(migraphx::make_op("add"), z, add1);
        m2.add_instruction(pass_op{}, add2);
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_split_between_add)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4}};
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", s);
        auto x     = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto sum = m1.add_instruction(migraphx::make_op("add"), x, y);
        m1.add_instruction(pass_op{}, sum);
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

static void test_dot_horiz(migraphx::shape::type_t type, const std::string& dot_type)
{
    auto s = migraphx::shape{type, {3, 2, 2}};
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", s);
        auto a     = m1.add_literal(migraphx::generate_literal(s, 0));
        auto b     = m1.add_literal(migraphx::generate_literal(s, 1));
        auto x     = m1.add_instruction(migraphx::make_op(dot_type), input, a);
        auto y     = m1.add_instruction(migraphx::make_op(dot_type), input, b);
        auto sum   = m1.add_instruction(migraphx::make_op("add"), x, y);
        m1.add_instruction(pass_op{}, sum);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto input  = m2.add_parameter("input", s);
        auto a      = m2.add_literal(migraphx::generate_literal(s, 0));
        auto b      = m2.add_literal(migraphx::generate_literal(s, 1));
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 2}}), a, b);
        auto dot    = m2.add_instruction(migraphx::make_op(dot_type), input, concat);
        auto x      = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {2}}}), dot);
        auto y = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {2}}, {"ends", {4}}}), dot);
        auto sum = m2.add_instruction(migraphx::make_op("add"), x, y);
        m2.add_instruction(pass_op{}, sum);
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_dot_horiz) { test_dot_horiz(migraphx::shape::int32_type, "dot"); }

TEST_CASE(simplify_quant_dot_horiz) { test_dot_horiz(migraphx::shape::int8_type, "quant_dot"); }

TEST_CASE(simplify_dot_horiz_same_constant)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 2}};
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", s);
        auto a     = m1.add_literal(migraphx::generate_literal(s, 0));
        auto x     = m1.add_instruction(migraphx::make_op("dot"), input, a);
        auto y     = m1.add_instruction(migraphx::make_op("dot"), input, a);
        auto sum   = m1.add_instruction(migraphx::make_op("add"), x, y);
        m1.add_instruction(pass_op{}, sum);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto input  = m2.add_parameter("input", s);
        auto a      = m2.add_literal(migraphx::generate_literal(s, 0));
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 2}}), a, a);
        auto dot    = m2.add_instruction(migraphx::make_op("dot"), input, concat);
        auto x      = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {2}}}), dot);
        auto y = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {2}}, {"ends", {4}}}), dot);
        auto sum = m2.add_instruction(migraphx::make_op("add"), x, y);
        m2.add_instruction(pass_op{}, sum);
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_dot_horiz_flipped)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 2}};
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", s);
        auto a     = m1.add_literal(migraphx::generate_literal(s, 0));
        auto b     = m1.add_literal(migraphx::generate_literal(s, 1));
        auto x     = m1.add_instruction(migraphx::make_op("dot"), input, a);
        auto y     = m1.add_instruction(migraphx::make_op("dot"), b, input);
        auto sum   = m1.add_instruction(migraphx::make_op("add"), x, y);
        m1.add_instruction(pass_op{}, sum);
    }

    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

// test if contiguous is added as necessary for reshapes
TEST_CASE(simplify_dot_horiz_reshape)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 4, 4}};
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", s);
        auto a     = m1.add_literal(migraphx::generate_literal(s, 0));
        auto b     = m1.add_literal(migraphx::generate_literal(s, 1));
        auto x     = m1.add_instruction(migraphx::make_op("dot"), input, a);
        auto y     = m1.add_instruction(migraphx::make_op("dot"), input, b);
        auto x_rsp = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 4, 2, 2}}}), x);
        auto y_rsp =
            m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}, {"steps", {2}}}), y);
        auto sum = m1.add_instruction(migraphx::make_op("add"), {x_rsp, y_rsp});
        m1.add_instruction(pass_op{}, sum);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto input  = m2.add_parameter("input", s);
        auto a      = m2.add_literal(migraphx::generate_literal(s, 0));
        auto b      = m2.add_literal(migraphx::generate_literal(s, 1));
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 2}}), a, b);
        auto dot    = m2.add_instruction(migraphx::make_op("dot"), input, concat);
        auto x      = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {4}}}), dot);
        auto y = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {4}}, {"ends", {8}}}), dot);
        auto x_rsp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 4, 2, 2}}}), x);
        auto y_rsp =
            m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}, {"steps", {2}}}), y);
        auto sum = m2.add_instruction(migraphx::make_op("add"), {x_rsp, y_rsp});
        m2.add_instruction(pass_op{}, sum);
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_conv_horiz)
{
    auto s  = migraphx::shape{migraphx::shape::int32_type, {8, 3, 64, 64}};
    auto ws = migraphx::shape{migraphx::shape::int32_type, {12, 3, 3, 3}};
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", s);
        auto a     = m1.add_literal(migraphx::generate_literal(ws, 0));
        auto b     = m1.add_literal(migraphx::generate_literal(ws, 1));
        auto x     = m1.add_instruction(migraphx::make_op("convolution"), input, a);
        auto y     = m1.add_instruction(migraphx::make_op("convolution"), input, b);
        auto sum   = m1.add_instruction(migraphx::make_op("add"), x, y);
        m1.add_instruction(pass_op{}, sum);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto input  = m2.add_parameter("input", s);
        auto a      = m2.add_literal(migraphx::generate_literal(ws, 0));
        auto b      = m2.add_literal(migraphx::generate_literal(ws, 1));
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), a, b);
        auto conv   = m2.add_instruction(migraphx::make_op("convolution"), input, concat);
        auto x      = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {12}}}), conv);
        auto y = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {12}}, {"ends", {24}}}), conv);
        auto sum = m2.add_instruction(migraphx::make_op("add"), x, y);
        m2.add_instruction(pass_op{}, sum);
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_group_conv_horiz)
{
    auto s  = migraphx::shape{migraphx::shape::int32_type, {1, 32, 111, 111}};
    auto ws = migraphx::shape{migraphx::shape::int32_type, {32, 1, 7, 7}};
    migraphx::module m1;
    {
        auto x     = m1.add_parameter("x", s);
        auto w1    = m1.add_literal(migraphx::generate_literal(ws, 1));
        auto w2    = m1.add_literal(migraphx::generate_literal(ws, 2));
        auto conv1 = m1.add_instruction(
            migraphx::make_op(
                "convolution",
                {{"padding", {3, 3}}, {"stride", {2, 2}}, {"dilation", {1, 1}}, {"group", 32}}),
            x,
            w1);
        auto conv2 = m1.add_instruction(
            migraphx::make_op(
                "convolution",
                {{"padding", {3, 3}}, {"stride", {2, 2}}, {"dilation", {1, 1}}, {"group", 32}}),
            x,
            w2);
        m1.add_instruction(pass_op{}, conv1, conv2);
    }
    migraphx::module m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_conv_horiz_grouped)
{
    auto s   = migraphx::shape{migraphx::shape::int32_type, {8, 6, 64, 64}};
    auto ws1 = migraphx::shape{migraphx::shape::int32_type, {6, 6, 3, 3}};
    auto ws2 = migraphx::shape{migraphx::shape::int32_type, {8, 6, 64, 64}};
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", s);
        auto a     = m1.add_literal(migraphx::generate_literal(ws1, 0));
        auto b     = m1.add_literal(migraphx::generate_literal(ws1, 1));
        auto c     = m1.add_literal(migraphx::generate_literal(ws2, 2));
        auto d     = m1.add_literal(migraphx::generate_literal(ws2, 3));
        auto convx =
            m1.add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}}), input, a);
        auto convy =
            m1.add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}}), input, b);
        auto dotx = m1.add_instruction(migraphx::make_op("dot"), input, c);
        auto doty = m1.add_instruction(migraphx::make_op("dot"), input, d);
        auto sum1 = m1.add_instruction(migraphx::make_op("add"), convx, convy);
        auto sum2 = m1.add_instruction(migraphx::make_op("add"), dotx, doty);
        auto sum3 = m1.add_instruction(migraphx::make_op("add"), sum1, sum2);

        m1.add_instruction(pass_op{}, sum3);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto input   = m2.add_parameter("input", s);
        auto a       = m2.add_literal(migraphx::generate_literal(ws1, 0));
        auto b       = m2.add_literal(migraphx::generate_literal(ws1, 1));
        auto c       = m2.add_literal(migraphx::generate_literal(ws2, 2));
        auto d       = m2.add_literal(migraphx::generate_literal(ws2, 3));
        auto concat1 = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), a, b);
        auto concat2 = m2.add_instruction(migraphx::make_op("concat", {{"axis", 3}}), c, d);
        auto conv    = m2.add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1}}}), input, concat1);
        auto convx = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {6}}}), conv);
        auto convy = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {6}}, {"ends", {12}}}), conv);
        auto sum1 = m2.add_instruction(migraphx::make_op("add"), convx, convy);
        auto dot  = m2.add_instruction(migraphx::make_op("dot"), input, concat2);
        auto dotx = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {64}}}), dot);
        auto doty = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {64}}, {"ends", {128}}}), dot);
        auto sum2 = m2.add_instruction(migraphx::make_op("add"), dotx, doty);
        auto sum3 = m2.add_instruction(migraphx::make_op("add"), sum1, sum2);
        m2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_conv_horiz_grouped_extra1)
{
    auto s   = migraphx::shape{migraphx::shape::int32_type, {8, 6, 64, 64}};
    auto ws1 = migraphx::shape{migraphx::shape::int32_type, {6, 6, 3, 3}};
    auto ws2 = migraphx::shape{migraphx::shape::int32_type, {8, 6, 64, 64}};
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", s);
        auto a     = m1.add_literal(migraphx::generate_literal(ws1, 0));
        auto b     = m1.add_literal(migraphx::generate_literal(ws1, 1));
        auto c     = m1.add_literal(migraphx::generate_literal(ws2, 2));
        auto d     = m1.add_literal(migraphx::generate_literal(ws2, 3));
        auto e     = m1.add_literal(migraphx::generate_literal(s, 4));
        auto convx =
            m1.add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}}), input, a);
        auto convy =
            m1.add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}}), input, b);
        auto dotx    = m1.add_instruction(migraphx::make_op("dot"), input, c);
        auto doty    = m1.add_instruction(migraphx::make_op("dot"), input, d);
        auto sqdiffx = m1.add_instruction(migraphx::make_op("sqdiff"), input, e);
        auto sum1    = m1.add_instruction(migraphx::make_op("add"), convx, convy);
        auto sum2    = m1.add_instruction(migraphx::make_op("add"), dotx, doty);
        auto sum3    = sqdiffx;
        auto sum4    = m1.add_instruction(migraphx::make_op("add"), sum1, sum2);
        auto sum5    = m1.add_instruction(migraphx::make_op("add"), sum4, sum3);
        m1.add_instruction(pass_op{}, sum5);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto input   = m2.add_parameter("input", s);
        auto a       = m2.add_literal(migraphx::generate_literal(ws1, 0));
        auto b       = m2.add_literal(migraphx::generate_literal(ws1, 1));
        auto c       = m2.add_literal(migraphx::generate_literal(ws2, 2));
        auto d       = m2.add_literal(migraphx::generate_literal(ws2, 3));
        auto e       = m2.add_literal(migraphx::generate_literal(s, 4));
        auto concat1 = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), a, b);
        auto concat2 = m2.add_instruction(migraphx::make_op("concat", {{"axis", 3}}), c, d);
        auto conv    = m2.add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1}}}), input, concat1);
        auto convx = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {6}}}), conv);
        auto convy = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {6}}, {"ends", {12}}}), conv);
        auto sum1 = m2.add_instruction(migraphx::make_op("add"), convx, convy);
        auto dot  = m2.add_instruction(migraphx::make_op("dot"), input, concat2);
        auto dotx = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {64}}}), dot);
        auto doty = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {64}}, {"ends", {128}}}), dot);
        auto sum2    = m2.add_instruction(migraphx::make_op("add"), dotx, doty);
        auto sqdiffx = m2.add_instruction(migraphx::make_op("sqdiff"), input, e);
        auto sum3    = sqdiffx;
        auto sum4    = m2.add_instruction(migraphx::make_op("add"), sum1, sum2);
        auto sum5    = m2.add_instruction(migraphx::make_op("add"), sum4, sum3);
        m2.add_instruction(pass_op{}, sum5);
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_conv_horiz_grouped_extra2)
{
    auto s   = migraphx::shape{migraphx::shape::int32_type, {8, 6, 64, 64}};
    auto ws1 = migraphx::shape{migraphx::shape::int32_type, {6, 6, 3, 3}};
    auto ws2 = migraphx::shape{migraphx::shape::int32_type, {8, 6, 64, 64}};
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", s);
        auto a     = m1.add_literal(migraphx::generate_literal(ws1, 0));
        auto b     = m1.add_literal(migraphx::generate_literal(ws1, 1));
        auto c     = m1.add_literal(migraphx::generate_literal(ws2, 2));
        auto d     = m1.add_literal(migraphx::generate_literal(ws2, 3));
        auto e     = m1.add_literal(migraphx::generate_literal(s, 4));
        auto f     = m1.add_literal(migraphx::generate_literal(s, 5));
        auto convx =
            m1.add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}}), input, a);
        auto convy =
            m1.add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}}), input, b);
        auto dotx    = m1.add_instruction(migraphx::make_op("dot"), input, c);
        auto doty    = m1.add_instruction(migraphx::make_op("dot"), input, d);
        auto sqdiffx = m1.add_instruction(migraphx::make_op("sqdiff"), input, e);
        auto sqdiffy = m1.add_instruction(migraphx::make_op("sqdiff"), input, f);
        auto sum1    = m1.add_instruction(migraphx::make_op("add"), convx, convy);
        auto sum2    = m1.add_instruction(migraphx::make_op("add"), dotx, doty);
        auto sum3    = m1.add_instruction(migraphx::make_op("add"), sqdiffx, sqdiffy);
        auto sum4    = m1.add_instruction(migraphx::make_op("add"), sum1, sum2);
        auto sum5    = m1.add_instruction(migraphx::make_op("add"), sum4, sum3);
        m1.add_instruction(pass_op{}, sum5);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto input   = m2.add_parameter("input", s);
        auto a       = m2.add_literal(migraphx::generate_literal(ws1, 0));
        auto b       = m2.add_literal(migraphx::generate_literal(ws1, 1));
        auto c       = m2.add_literal(migraphx::generate_literal(ws2, 2));
        auto d       = m2.add_literal(migraphx::generate_literal(ws2, 3));
        auto e       = m2.add_literal(migraphx::generate_literal(s, 4));
        auto f       = m2.add_literal(migraphx::generate_literal(s, 5));
        auto concat1 = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), a, b);
        auto concat2 = m2.add_instruction(migraphx::make_op("concat", {{"axis", 3}}), c, d);
        auto conv    = m2.add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1}}}), input, concat1);
        auto convx = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {6}}}), conv);
        auto convy = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {6}}, {"ends", {12}}}), conv);
        auto sum1 = m2.add_instruction(migraphx::make_op("add"), convx, convy);
        auto dot  = m2.add_instruction(migraphx::make_op("dot"), input, concat2);
        auto dotx = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {64}}}), dot);
        auto doty = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {64}}, {"ends", {128}}}), dot);
        auto sum2    = m2.add_instruction(migraphx::make_op("add"), dotx, doty);
        auto sqdiffx = m2.add_instruction(migraphx::make_op("sqdiff"), input, e);
        auto sqdiffy = m2.add_instruction(migraphx::make_op("sqdiff"), input, f);
        auto sum3    = m2.add_instruction(migraphx::make_op("add"), sqdiffx, sqdiffy);
        auto sum4    = m2.add_instruction(migraphx::make_op("add"), sum1, sum2);
        auto sum5    = m2.add_instruction(migraphx::make_op("add"), sum4, sum3);
        m2.add_instruction(pass_op{}, sum5);
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(simplify_mul_slice_conv_horiz_fusion)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::int32_type, {1, 1024, 17, 17}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::int32_type, {768, 1024, 1, 1}}));
        auto conv   = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {384}}}), conv);
        auto a1 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {384}}, 1));
        auto b1 = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 384, 17, 17}}}), a1);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), slice1, b1);
        auto a2 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {384}}, 2));
        auto b2 = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 384, 17, 17}}}), a2);
        auto add1 = m1.add_instruction(migraphx::make_op("add"), mul, b2);
        auto a3 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {384}}, 3));
        auto b3 = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 384, 17, 17}}}), a3);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {384}}, {"ends", {768}}}), conv);
        auto add2 = m1.add_instruction(migraphx::make_op("add"), slice2, b3);
        m1.add_instruction(pass_op{}, add1, add2);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::int32_type, {1, 1024, 17, 17}});
        auto w = m2.add_literal(
            migraphx::generate_literal({migraphx::shape::int32_type, {768, 1024, 1, 1}}));
        auto wslice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {384}}}), w);
        auto a1 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {384}}, 1));
        auto b1 = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {384, 1024, 1, 1}}}), a1);
        auto mul     = m2.add_instruction(migraphx::make_op("mul"), b1, wslice1);
        auto wslice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {384}}, {"ends", {768}}}), w);
        auto concat1 = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), mul, wslice2);
        auto conv    = m2.add_instruction(migraphx::make_op("convolution"), x, concat1);
        auto a2 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {384}}, 2));
        auto a3 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {384}}, 3));
        auto concat2 = m2.add_instruction(migraphx::make_op("concat"), a2, a3);
        auto b4      = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 768, 17, 17}}}), concat2);
        auto add    = m2.add_instruction(migraphx::make_op("add"), conv, b4);
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {384}}}), add);
        auto slice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {384}}, {"ends", {768}}}), add);
        m2.add_instruction(pass_op{}, slice1, slice2);
    }
    EXPECT(m1.sort() == m2.sort());
}

template <std::size_t BS, bool TransposeInput>
static void reorder_reshape_slice()
{
    std::vector<int64_t> perm0 = {0, 2, 1, 3};
    std::vector<int64_t> perm1 = {0, 2, 3, 1};
    migraphx::module m1;
    {
        auto s = migraphx::shape{migraphx::shape::float_type, {BS, 128, 1920}};
        if(TransposeInput)
        {
            s = migraphx::shape{migraphx::shape::float_type, {BS, 128, 1920}, {165120, 1, 128}};
        }
        auto input = m1.add_parameter("input", s);
        auto slc0  = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {640}}}), input);
        auto slc1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {640}}, {"ends", {1280}}}),
            input);
        auto slc2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1280}}, {"ends", {1920}}}),
            input);

        auto c0 = m1.add_instruction(migraphx::make_op("contiguous"), slc0);
        auto c1 = m1.add_instruction(migraphx::make_op("contiguous"), slc1);
        auto c2 = m1.add_instruction(migraphx::make_op("contiguous"), slc2);

        std::vector<int64_t> lens = {static_cast<int64_t>(BS), 128, 10, 64};
        auto r0 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c0);
        auto r1 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c1);
        auto r2 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c2);

        auto t0 = m1.add_instruction(migraphx::make_op("transpose", {{"permutation", perm0}}), r0);
        auto t1 = m1.add_instruction(migraphx::make_op("transpose", {{"permutation", perm0}}), r1);
        auto t2 = m1.add_instruction(migraphx::make_op("transpose", {{"permutation", perm1}}), r2);

        auto sum = m1.add_instruction(migraphx::make_op("add"), t0, t1);
        auto ret = m1.add_instruction(migraphx::make_op("dot"), sum, t2);
        m1.add_return({ret});
    };

    migraphx::module m2;
    {
        auto s = migraphx::shape{migraphx::shape::float_type, {BS, 128, 1920}};
        if(TransposeInput)
        {
            s = migraphx::shape{migraphx::shape::float_type, {BS, 128, 1920}, {165120, 1, 128}};
        }
        auto input                = m2.add_parameter("input", s);
        auto rsp_input            = input;
        std::vector<int64_t> lens = {static_cast<int64_t>(BS), 128, 30, 64};
        auto r = m2.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), rsp_input);

        auto slc0 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {10}}}), r);
        auto slc1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {10}}, {"ends", {20}}}), r);
        auto slc2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {20}}, {"ends", {30}}}), r);

        auto t0 =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", perm0}}), slc0);
        auto t1 =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", perm0}}), slc1);
        auto t2 =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", perm1}}), slc2);

        auto sum = m2.add_instruction(migraphx::make_op("add"), t0, t1);
        auto ret = m2.add_instruction(migraphx::make_op("dot"), sum, t2);
        m2.add_return({ret});
    };
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE_REGISTER(reorder_reshape_slice<1, true>); // test if contiguous is added as necessary if
                                                    // input is transposed
TEST_CASE_REGISTER(reorder_reshape_slice<4, true>);
TEST_CASE_REGISTER(reorder_reshape_slice<8, true>);
TEST_CASE_REGISTER(reorder_reshape_slice<1, false>);
TEST_CASE_REGISTER(reorder_reshape_slice<4, false>);
TEST_CASE_REGISTER(reorder_reshape_slice<8, false>);

template <std::size_t BS>
static void reorder_reshape_slice_move_axis1()
{
    migraphx::module m1;
    {
        auto s                     = migraphx::shape{migraphx::shape::float_type, {BS, 256, 96}};
        std::vector<int64_t> perm0 = {0, 2, 1, 3};
        std::vector<int64_t> perm1 = {0, 2, 3, 1};
        auto input                 = m1.add_parameter("input", s);
        auto slc0                  = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {32}}}), input);
        auto slc1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {32}}, {"ends", {64}}}), input);
        auto slc2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {64}}, {"ends", {96}}}), input);

        auto c0 = m1.add_instruction(migraphx::make_op("contiguous"), slc0);
        auto c1 = m1.add_instruction(migraphx::make_op("contiguous"), slc1);
        auto c2 = m1.add_instruction(migraphx::make_op("contiguous"), slc2);

        std::vector<int64_t> lens = {static_cast<int64_t>(BS), 64, 4, 32};
        auto r0 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c0);
        auto r1 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c1);
        auto r2 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c2);

        auto t0 = m1.add_instruction(migraphx::make_op("transpose", {{"permutation", perm0}}), r0);
        auto t1 = m1.add_instruction(migraphx::make_op("transpose", {{"permutation", perm0}}), r1);
        auto t2 = m1.add_instruction(migraphx::make_op("transpose", {{"permutation", perm1}}), r2);

        auto sum = m1.add_instruction(migraphx::make_op("add"), t0, t1);
        auto ret = m1.add_instruction(migraphx::make_op("dot"), sum, t2);
        m1.add_return({ret});
    };

    migraphx::module m2;
    {
        auto s                     = migraphx::shape{migraphx::shape::float_type, {BS, 256, 96}};
        std::vector<int64_t> perm0 = {0, 2, 1, 3};
        std::vector<int64_t> perm1 = {0, 2, 3, 1};
        auto input                 = m2.add_parameter("input", s);
        std::vector<int64_t> lens  = {static_cast<int64_t>(BS), 64, 4, 96};
        auto rsp  = m2.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), input);
        auto slc0 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {32}}}), rsp);
        auto t0 =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", perm0}}), slc0);
        auto slc1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {32}}, {"ends", {64}}}), rsp);
        auto t1 =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", perm0}}), slc1);
        auto slc2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {64}}, {"ends", {96}}}), rsp);
        auto t2 =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", perm1}}), slc2);

        auto sum = m2.add_instruction(migraphx::make_op("add"), t0, t1);
        auto ret = m2.add_instruction(migraphx::make_op("dot"), sum, t2);
        m2.add_return({ret});
    };

    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE_REGISTER(reorder_reshape_slice_move_axis1<4>);
TEST_CASE_REGISTER(reorder_reshape_slice_move_axis1<8>);

TEST_CASE(reorder_reshape_slice_move_axis2)
{
    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {128, 96}};
        auto input = m1.add_parameter("input", s);
        auto slc0  = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {32}}}), input);
        auto slc1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {32}}, {"ends", {64}}}), input);
        auto slc2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {64}}, {"ends", {96}}}), input);

        auto c0 = m1.add_instruction(migraphx::make_op("contiguous"), slc0);
        auto c1 = m1.add_instruction(migraphx::make_op("contiguous"), slc1);
        auto c2 = m1.add_instruction(migraphx::make_op("contiguous"), slc2);

        std::vector<int64_t> lens = {1, 16, 8, 32};
        auto r0 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c0);
        auto r1 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c1);
        auto r2 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c2);

        auto sum = m1.add_instruction(migraphx::make_op("add"), r0, r1);
        auto ret = m1.add_instruction(migraphx::make_op("mul"), sum, r2);
        m1.add_return({ret});
    };

    migraphx::module m2;
    {
        auto s                    = migraphx::shape{migraphx::shape::float_type, {128, 96}};
        auto input                = m2.add_parameter("input", s);
        std::vector<int64_t> lens = {1, 16, 8, 96};
        auto rsp  = m2.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), input);
        auto slc0 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {32}}}), rsp);
        auto slc1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {32}}, {"ends", {64}}}), rsp);
        auto slc2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {64}}, {"ends", {96}}}), rsp);

        auto sum = m2.add_instruction(migraphx::make_op("add"), slc0, slc1);
        auto ret = m2.add_instruction(migraphx::make_op("mul"), sum, slc2);
        m2.add_return({ret});
    };

    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reorder_reshape_slice_len_1)
{
    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {1, 128, 3}};
        auto input = m1.add_parameter("input", s);
        auto slc0  = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto slc1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto slc2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {2}}, {"ends", {3}}}), input);

        auto c0 = m1.add_instruction(migraphx::make_op("contiguous"), slc0);
        auto c1 = m1.add_instruction(migraphx::make_op("contiguous"), slc1);
        auto c2 = m1.add_instruction(migraphx::make_op("contiguous"), slc2);

        std::vector<int64_t> lens = {1, 128};
        auto r0 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c0);
        auto r1 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c1);
        auto r2 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c2);

        auto sum = m1.add_instruction(migraphx::make_op("add"), r0, r1);
        auto ret = m1.add_instruction(migraphx::make_op("mul"), sum, r2);
        m1.add_return({ret});
    };

    migraphx::module m2;
    {
        auto s                    = migraphx::shape{migraphx::shape::float_type, {1, 128, 3}};
        auto input                = m2.add_parameter("input", s);
        std::vector<int64_t> lens = {1, 384};
        auto rsp  = m2.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), input);
        auto slc0 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {128}}}), rsp);
        auto slc1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {128}}, {"ends", {256}}}), rsp);
        auto slc2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {256}}, {"ends", {384}}}), rsp);

        auto sum = m2.add_instruction(migraphx::make_op("add"), slc0, slc1);
        auto ret = m2.add_instruction(migraphx::make_op("mul"), sum, slc2);
        m2.add_return({ret});
    };

    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reorder_reshape_slice_not_apply)
{
    auto create_p = [] {
        migraphx::module m;
        migraphx::shape s{migraphx::shape::float_type, {128, 96}};
        auto input = m.add_parameter("input", s);
        auto slc0  = m.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {32}}}), input);
        auto slc1 = m.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {32}}, {"ends", {64}}}), input);
        auto slc2 = m.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {64}}, {"ends", {96}}}), input);

        auto c0 = m.add_instruction(migraphx::make_op("contiguous"), slc0);
        auto c1 = m.add_instruction(migraphx::make_op("contiguous"), slc1);
        auto c2 = m.add_instruction(migraphx::make_op("contiguous"), slc2);

        std::vector<int64_t> lens = {1, 16, 16, 16};
        auto r0 = m.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c0);
        auto r1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c1);
        auto r2 = m.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c2);

        auto sum = m.add_instruction(migraphx::make_op("add"), r0, r1);
        auto ret = m.add_instruction(migraphx::make_op("mul"), sum, r2);
        m.add_return({ret});

        return m;
    };

    auto m1 = create_p();
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reorder_reshape_slice_multi_rsp)
{
    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {4, 128, 3, 32, 80}};
        auto input = m1.add_parameter("input", s);
        auto t1    = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {2, 0, 3, 1, 4}}}), input);
        auto slc0 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), t1);
        auto slc1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), t1);
        auto slc2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), t1);

        auto c1_1 = m1.add_instruction(migraphx::make_op("contiguous"), slc1);
        auto c2_1 = m1.add_instruction(migraphx::make_op("contiguous"), slc2);

        auto c1 = m1.add_instruction(migraphx::make_op("contiguous"), slc1);
        auto r1 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 32, 128, 80}}}), c1);

        auto c2 = m1.add_instruction(migraphx::make_op("contiguous"), slc2);
        auto r2 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 32, 128, 80}}}), c2);

        auto r1_1 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {128, 128, 80}}}), c1_1);
        auto r2_1 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {128, 128, 80}}}), c2_1);

        auto c0 = m1.add_instruction(migraphx::make_op("contiguous"), slc0);
        auto r0 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {128, 128, 80}}}), c0);

        auto t2 =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), r1_1);
        auto c_t2 = m1.add_instruction(migraphx::make_op("contiguous"), t2);

        auto dot = m1.add_instruction(migraphx::make_op("dot"), r0, c_t2);

        m1.add_return({r1, r2, dot, r2_1});
    };

    migraphx::module m2;
    {
        migraphx::shape s{migraphx::shape::float_type, {4, 128, 3, 32, 80}};
        auto input = m2.add_parameter("input", s);
        auto t1    = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {2, 0, 3, 1, 4}}}), input);
        auto rsp1 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {384, 128, 80}}}), t1);
        auto slc0 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {256}}, {"ends", {384}}}), rsp1);
        auto slc1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {128}}, {"ends", {256}}}), rsp1);

        auto t_slc1 =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), slc1);
        auto c_t_slc1 = m2.add_instruction(migraphx::make_op("contiguous"), t_slc1);

        auto slc2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {128}}}), rsp1);

        auto dot = m2.add_instruction(migraphx::make_op("dot"), slc2, c_t_slc1);

        auto rsp2 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {12, 32, 128, 80}}}), t1);

        auto slc2_1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {8}}}), rsp2);

        auto slc2_2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {8}}, {"ends", {12}}}), rsp2);

        m2.add_return({slc2_1, slc2_2, dot, slc0});
    };

    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reorder_reshape_slice_partial)
{
    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {128, 96}};
        auto input = m1.add_parameter("input", s);
        auto slc0  = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {8}}}), input);
        auto slc1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {8}}, {"ends", {16}}}), input);
        auto slc2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {16}}, {"ends", {24}}}), input);
        auto slc3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {24}}, {"ends", {128}}}), input);

        auto c0 = m1.add_instruction(migraphx::make_op("contiguous"), slc0);
        auto c1 = m1.add_instruction(migraphx::make_op("contiguous"), slc1);
        auto c2 = m1.add_instruction(migraphx::make_op("contiguous"), slc2);

        std::vector<int64_t> lens = {2, 4, 96};
        auto r0 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c0);
        auto r1 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c1);
        auto r2 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c2);

        auto sum = m1.add_instruction(migraphx::make_op("add"), r0, r1);
        auto ret = m1.add_instruction(migraphx::make_op("mul"), sum, r2);
        m1.add_return({ret, slc3});
    };

    migraphx::module m2;
    {
        migraphx::shape s{migraphx::shape::float_type, {128, 96}};
        auto input = m2.add_parameter("input", s);
        auto rsp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {32, 4, 96}}}), input);
        auto slc3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {24}}, {"ends", {128}}}), input);

        auto slc0 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {2}}}), rsp);
        auto slc1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {4}}}), rsp);
        auto slc2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {6}}}), rsp);

        auto sum = m2.add_instruction(migraphx::make_op("add"), slc0, slc1);
        auto ret = m2.add_instruction(migraphx::make_op("mul"), sum, slc2);
        m2.add_return({ret, slc3});
    };

    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reorder_reshape_slice_uneven_slice)
{
    auto create_p = [] {
        migraphx::module m;
        migraphx::shape s{migraphx::shape::float_type, {128, 96}};
        auto input = m.add_parameter("input", s);
        auto slc0  = m.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {31}}}), input);
        auto slc1 = m.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {31}}, {"ends", {62}}}), input);
        auto slc2 = m.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {62}}, {"ends", {93}}}), input);
        auto slc3 = m.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {93}}, {"ends", {128}}}), input);

        auto c0 = m.add_instruction(migraphx::make_op("contiguous"), slc0);
        auto c1 = m.add_instruction(migraphx::make_op("contiguous"), slc1);
        auto c2 = m.add_instruction(migraphx::make_op("contiguous"), slc2);

        std::vector<int64_t> lens = {1, 31, 96};
        auto r0 = m.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c0);
        auto r1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c1);
        auto r2 = m.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c2);

        auto sum = m.add_instruction(migraphx::make_op("add"), r0, r1);
        auto ret = m.add_instruction(migraphx::make_op("mul"), sum, r2);
        m.add_return({ret, slc3});

        return m;
    };

    auto m1 = create_p();
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

template <std::size_t BS>
static void reorder_reshape_slice_diff_dims()
{
    migraphx::module m1;
    {
        auto s     = migraphx::shape{migraphx::shape::float_type, {BS, 96, 96}};
        auto input = m1.add_parameter("input", s);
        auto slc0  = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {32}}}), input);
        auto slc1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {32}}, {"ends", {64}}}), input);
        auto slc2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {64}}, {"ends", {96}}}), input);

        auto c0 = m1.add_instruction(migraphx::make_op("contiguous"), slc0);
        auto c1 = m1.add_instruction(migraphx::make_op("contiguous"), slc1);
        auto c2 = m1.add_instruction(migraphx::make_op("contiguous"), slc2);

        std::vector<int64_t> lens  = {static_cast<int64_t>(BS), 32, 3, 32};
        std::vector<int64_t> lens1 = {static_cast<int64_t>(BS), 48, 2, 32};
        auto r0 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c0);
        auto r1 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", lens1}}), c1);
        auto r2 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c2);

        m1.add_return({r0, r1, r2});
    };

    migraphx::module m2;
    {
        auto s     = migraphx::shape{migraphx::shape::float_type, {BS, 96, 96}};
        auto input = m2.add_parameter("input", s);
        auto slc1  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {32}}, {"ends", {64}}}), input);
        auto c1                    = m2.add_instruction(migraphx::make_op("contiguous"), slc1);
        std::vector<int64_t> lens1 = {static_cast<int64_t>(BS), 48, 2, 32};
        auto r1 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", lens1}}), c1);

        std::vector<int64_t> lens = {static_cast<int64_t>(BS), 32, 3, 96};
        auto r_new = m2.add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), input);
        auto slc0  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {32}}}), r_new);
        auto slc2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {64}}, {"ends", {96}}}), r_new);

        m2.add_return({slc0, r1, slc2});
    };

    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE_REGISTER(reorder_reshape_slice_diff_dims<4>);
TEST_CASE_REGISTER(reorder_reshape_slice_diff_dims<8>);

template <std::size_t BS>
static void reorder_slice_trans()
{
    std::vector<int64_t> perm = {0, 2, 1};

    migraphx::module m1;
    {
        auto s     = migraphx::shape{migraphx::shape::float_type, {BS, 128, 1920}};
        auto input = m1.add_parameter("input", s);
        auto slc0  = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {640}}}), input);
        auto slc1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {640}}, {"ends", {1280}}}),
            input);
        auto slc2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1280}}, {"ends", {1920}}}),
            input);

        auto t0 = m1.add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), slc0);
        auto t1 = m1.add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), slc1);
        auto t2 = m1.add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), slc2);

        auto sum = m1.add_instruction(migraphx::make_op("add"), t0, t1);
        auto ret = m1.add_instruction(migraphx::make_op("mul"), sum, t2);
        m1.add_return({ret});
    };

    migraphx::module m2;
    {
        auto s     = migraphx::shape{migraphx::shape::float_type, {BS, 128, 1920}};
        auto input = m2.add_parameter("input", s);
        auto r = m2.add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), input);

        auto slc0 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {640}}}), r);
        auto slc1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {640}}, {"ends", {1280}}}), r);
        auto slc2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1280}}, {"ends", {1920}}}), r);

        auto sum = m2.add_instruction(migraphx::make_op("add"), slc0, slc1);
        auto ret = m2.add_instruction(migraphx::make_op("mul"), sum, slc2);
        m2.add_return({ret});
    };

    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE_REGISTER(reorder_slice_trans<1>);
TEST_CASE_REGISTER(reorder_slice_trans<8>);

template <std::size_t BS>
static void reorder_slice_trans_diff_perm()
{
    migraphx::module m1;
    {
        auto s                     = migraphx::shape{migraphx::shape::float_type, {BS, 128, 1920}};
        std::vector<int64_t> perm0 = {0, 2, 1};
        std::vector<int64_t> perm1 = {0, 1, 2};
        auto input                 = m1.add_parameter("input", s);
        auto slc0                  = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {640}}}), input);
        auto slc1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {640}}, {"ends", {1280}}}),
            input);
        auto slc2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1280}}, {"ends", {1920}}}),
            input);

        auto t0 =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", perm0}}), slc0);
        auto t1 =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", perm0}}), slc1);
        auto t2 =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", perm1}}), slc2);

        auto sum = m1.add_instruction(migraphx::make_op("add"), t0, t1);
        auto ret = m1.add_instruction(migraphx::make_op("dot"), sum, t2);
        m1.add_return({ret});
    };

    run_pass(m1);
    auto m2 = m1;
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE_REGISTER(reorder_slice_trans_diff_perm<1>);
TEST_CASE_REGISTER(reorder_slice_trans_diff_perm<4>);

TEST_CASE(reorder_slice_trans_multi_outputs)
{
    migraphx::module m1;
    {
        auto s                    = migraphx::shape{migraphx::shape::float_type, {8, 128, 1920}};
        auto input                = m1.add_parameter("input", s);
        std::vector<int64_t> perm = {0, 2, 1};
        auto slc0                 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {640}}}), input);
        auto slc1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {640}}, {"ends", {1280}}}),
            input);
        auto slc2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1280}}, {"ends", {1920}}}),
            input);

        auto t0 = m1.add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), slc0);
        auto t1 = m1.add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), slc1);
        auto t2 = m1.add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), slc2);

        auto sum      = m1.add_instruction(migraphx::make_op("add"), t0, t1);
        auto dot      = m1.add_instruction(migraphx::make_op("mul"), sum, t2);
        auto slc_cont = m1.add_instruction(migraphx::make_op("contiguous"), slc1);
        m1.add_return({slc_cont, dot});
    };
    run_pass(m1);
    auto m2 = m1;
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reorder_slice_ins_deps)
{
    auto create_module = [] {
        migraphx::module m;
        migraphx::shape sx{migraphx::shape::float_type, {4, 2}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 2}};
        std::vector<float> datax = {0, 1, 2, 3, 4, 5, 6, 7};
        std::vector<float> datay = {0, 1, 2, 3};
        auto inx                 = m.add_literal(migraphx::literal(sx, datax));
        auto iny                 = m.add_literal(migraphx::literal(sy, datay));
        auto slc0                = m.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {2}}}), inx);
        auto slc1 = m.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {4}}}), inx);
        auto n0 = m.add_instruction(migraphx::make_op("neg"), slc0);
        auto a0 = m.add_instruction(migraphx::make_op("add"), n0, slc1);
        auto m0 = m.add_instruction(migraphx::make_op("mul"), a0, iny);
        auto r  = m.add_instruction(migraphx::make_op("add"), m0, slc0);
        m.add_return({r});

        return m;
    };

    auto m = create_module();
    run_pass(m);
    EXPECT(m == create_module());
}

TEST_CASE(dot_broadcast_different_broadcast1)
{
    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", {migraphx::shape::float_type, {64}});
        auto y  = m1.add_parameter("y", {migraphx::shape::float_type, {64, 64}});
        auto xb = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 3}, {"out_lens", {2, 4, 4, 64}}}), x);
        auto yb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 64, 64}}}), y);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), xb, yb);
        m1.add_return({dot});
    };

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", {migraphx::shape::float_type, {64}});
        auto y  = m2.add_parameter("y", {migraphx::shape::float_type, {64, 64}});
        auto xb = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {4, 64}}}), x);
        auto dot       = m2.add_instruction(migraphx::make_op("dot"), xb, y);
        auto broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4, 64}}}), dot);
        m2.add_return({broadcast});
    };
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(dot_broadcast_different_broadcast2)
{
    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", {migraphx::shape::float_type, {384}});
        auto y  = m1.add_parameter("y", {migraphx::shape::float_type, {768, 3072}});
        auto xb = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 384, 768}}}), x);
        auto yb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 768, 3072}}}), y);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), xb, yb);
        m1.add_return({dot});
    };

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", {migraphx::shape::float_type, {384}});
        auto y  = m2.add_parameter("y", {migraphx::shape::float_type, {768, 3072}});
        auto xb = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {384, 768}}}), x);
        auto dot       = m2.add_instruction(migraphx::make_op("dot"), xb, y);
        auto broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 384, 3072}}}), dot);
        m2.add_return({broadcast});
    };

    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(dot_broadcast_different_rank)
{
    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", {migraphx::shape::float_type, {768}});
        auto y  = m1.add_parameter("y", {migraphx::shape::float_type, {768, 3072}});
        auto xb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 384, 768}}}), x);
        auto yb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 768, 3072}}}), y);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), xb, yb);
        m1.add_return({dot});
    };

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {768}});
        auto y = m2.add_parameter("y", {migraphx::shape::float_type, {768, 3072}});
        auto xb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {384, 768}}}), x);
        auto dot       = m2.add_instruction(migraphx::make_op("dot"), xb, y);
        auto broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 384, 3072}}}), dot);
        m2.add_return({broadcast});
    };

    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(dot_broadcast_unsqueezed_input)
{
    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", {migraphx::shape::float_type, {1, 1, 1, 8}});
        auto y  = m1.add_parameter("y", {migraphx::shape::float_type, {8, 8}});
        auto xb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 8, 8}}}), x);
        auto yb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 8, 8}}}), y);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), xb, yb);
        m1.add_return({dot});
    };

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", {migraphx::shape::float_type, {1, 1, 1, 8}});
        auto y    = m2.add_parameter("y", {migraphx::shape::float_type, {8, 8}});
        auto x_sq = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 1}}}), x);
        auto xb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8, 8}}}), x_sq);
        auto dot       = m2.add_instruction(migraphx::make_op("dot"), xb, y);
        auto broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 8, 8}}}), dot);
        m2.add_return({broadcast});
    };

    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(dot_fusion_reshape)
{
    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 4096, 320}};
        auto input = m1.add_parameter("input", s);

        auto p0 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {2, 320, 320}}, 0));
        auto p1 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {2, 320, 320}}, 1));

        auto d0 = m1.add_instruction(migraphx::make_op("dot"), input, p0);
        auto d1 = m1.add_instruction(migraphx::make_op("dot"), input, p1);

        auto r0 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 4096, 8, 40}}}), d0);
        m1.add_return({r0, d1});
    };

    migraphx::module m2;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 4096, 320}};
        auto input = m2.add_parameter("input", s);

        auto p0 = m2.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {2, 320, 320}}, 0));
        auto p1 = m2.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {2, 320, 320}}, 1));

        auto c = m2.add_instruction(migraphx::make_op("concat", {{"axis", 2}}), p0, p1);
        auto d = m2.add_instruction(migraphx::make_op("dot"), input, c);

        auto s0 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {320}}}), d);
        auto s1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {320}}, {"ends", {640}}}), d);

        auto r0 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 4096, 8, 40}}}), s0);

        m2.add_return({r0, s1});
    };

    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(mul_dot_a)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 256, 32}};
    migraphx::shape bs{migraphx::shape::float_type, {2, 32, 128}};
    migraphx::module m1;
    {
        auto a = m1.add_parameter("input", as);

        auto lit =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {1, 1, 32}}));
        auto litb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", as.lens()}}), lit);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), a, litb);

        auto b   = m1.add_literal(migraphx::generate_literal(bs));
        auto dot = m1.add_instruction(migraphx::make_op("dot"), mul, b);

        m1.add_return({dot});
    };
    run_pass(m1);

    migraphx::module m2;
    {
        auto a = m2.add_parameter("input", as);

        auto lit =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {1, 1, 32}}));
        auto litb = m2.add_instruction(
            migraphx::make_op("multibroadcast",
                              {{"out_lens", migraphx::reorder_dims(bs.lens(), {0, 2, 1})}}),
            lit);
        auto litt =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), litb);

        auto b   = m2.add_literal(migraphx::generate_literal(bs));
        auto mul = m2.add_instruction(migraphx::make_op("mul"), b, litt);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), a, mul);

        m2.add_return({dot});
    };

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(mul_dot_b)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 256, 32}};
    migraphx::shape bs{migraphx::shape::float_type, {2, 32, 128}};
    migraphx::module m1;
    {
        auto b = m1.add_parameter("input", bs);

        auto lit =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {1, 32, 1}}));
        auto litb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", bs.lens()}}), lit);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), b, litb);

        auto a   = m1.add_literal(migraphx::generate_literal(as));
        auto dot = m1.add_instruction(migraphx::make_op("dot"), a, mul);

        m1.add_return({dot});
    };
    run_pass(m1);

    migraphx::module m2;
    {

        auto b = m2.add_parameter("input", bs);

        auto lit =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {1, 32, 1}}));
        auto litb = m2.add_instruction(
            migraphx::make_op("multibroadcast",
                              {{"out_lens", migraphx::reorder_dims(as.lens(), {0, 2, 1})}}),
            lit);
        auto litt =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), litb);

        auto a   = m2.add_literal(migraphx::generate_literal(as));
        auto mul = m2.add_instruction(migraphx::make_op("mul"), a, litt);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), mul, b);

        m2.add_return({dot});
    };

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(mul_dot_a_not_k_broadcast)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 256, 32}};
    migraphx::shape bs{migraphx::shape::float_type, {2, 32, 128}};
    migraphx::module m1;
    {
        auto a = m1.add_parameter("input", as);

        auto lit =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {1, 256, 1}}));
        auto litb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", as.lens()}}), lit);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), a, litb);

        auto b   = m1.add_literal(migraphx::generate_literal(bs));
        auto dot = m1.add_instruction(migraphx::make_op("dot"), mul, b);

        m1.add_return({dot});
    };
    migraphx::module m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(mul_dot_b_not_k_broadcast)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 256, 32}};
    migraphx::shape bs{migraphx::shape::float_type, {2, 32, 128}};
    migraphx::module m1;
    {
        auto b = m1.add_parameter("input", bs);

        auto lit =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {1, 1, 128}}));
        auto litb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", bs.lens()}}), lit);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), b, litb);

        auto a   = m1.add_literal(migraphx::generate_literal(as));
        auto dot = m1.add_instruction(migraphx::make_op("dot"), a, mul);

        m1.add_return({dot});
    };
    migraphx::module m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(mul_dot_a_int4_dq)
{
    migraphx::shape as{migraphx::shape::float_type, {1, 32, 4096}};
    migraphx::shape bs{migraphx::shape::int8_type, {22016, 2048}};
    migraphx::shape cs{migraphx::shape::float_type, {22016, 4096}};
    migraphx::module m1;
    {
        auto a = m1.add_parameter("input", as);

        auto lit =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4096}}));
        auto litb = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 2}, {"out_lens", as.lens()}}), lit);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), a, litb);

        auto b         = m1.add_literal(migraphx::generate_literal(bs));
        auto unpack    = m1.add_instruction(migraphx::make_op("unpack_int4"), b);
        auto scales    = m1.add_literal(migraphx::generate_literal(cs));
        auto dq        = m1.add_instruction(migraphx::make_op("dequantizelinear"), unpack, scales);
        auto unsqueeze = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), dq);
        auto transpose = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), unsqueeze);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), mul, transpose);
        m1.add_return({dot});
    };
    migraphx::module m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(mul_dot_a_int4_dq_concat)
{
    migraphx::shape as{migraphx::shape::float_type, {1, 32, 4096}};
    migraphx::shape bs{migraphx::shape::int8_type, {4096, 5504}};
    migraphx::shape cs{migraphx::shape::float_type, {4096, 11008}};
    migraphx::module m1;
    {
        auto a = m1.add_parameter("input", as);

        auto lit =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4096}}));
        auto litb = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 2}, {"out_lens", as.lens()}}), lit);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), a, litb);

        std::vector<migraphx::instruction_ref> concats;
        for(int i = 0; i < 2; i++)
        {
            auto b      = m1.add_literal(migraphx::generate_literal(bs));
            auto unpack = m1.add_instruction(migraphx::make_op("unpack_int4"), b);
            auto scales = m1.add_literal(migraphx::generate_literal(cs));
            auto dq     = m1.add_instruction(migraphx::make_op("dequantizelinear"), unpack, scales);
            concats.push_back(
                m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), dq));
        }
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 2}}), concats);
        auto dot    = m1.add_instruction(migraphx::make_op("dot"), mul, concat);
        m1.add_return({dot});
    };
    migraphx::module m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(dot_mul_a)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 256, 32}};
    migraphx::shape bs{migraphx::shape::float_type, {2, 32, 128}};
    migraphx::module m1;
    {
        auto a   = m1.add_parameter("input", as);
        auto b   = m1.add_literal(migraphx::generate_literal(bs));
        auto dot = m1.add_instruction(migraphx::make_op("dot"), a, b);
        auto lit =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {1, 1, 128}}));
        auto litb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", dot->get_shape().lens()}}), lit);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), dot, litb);
        m1.add_return({mul});
    };
    run_pass(m1);

    migraphx::module m2;
    {
        auto a = m2.add_parameter("input", as);
        auto b = m2.add_literal(migraphx::generate_literal(bs));
        auto lit =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {1, 1, 128}}));
        auto litb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", bs.lens()}}), lit);

        auto mul = m2.add_instruction(migraphx::make_op("mul"), b, litb);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), a, mul);
        m2.add_return({dot});
    };

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(dot_mul_a_non_const)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 256, 32}};
    migraphx::shape bs{migraphx::shape::float_type, {2, 32, 128}};
    migraphx::module m1;
    {
        auto a   = m1.add_parameter("input", as);
        auto b   = m1.add_literal(migraphx::generate_literal(bs));
        auto dot = m1.add_instruction(migraphx::make_op("dot"), a, b);
        auto lit =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {1, 256, 1}}));
        auto litb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", dot->get_shape().lens()}}), lit);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), dot, litb);
        m1.add_return({mul});
    };
    migraphx::module m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(dot_mul_b)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 256, 32}};
    migraphx::shape bs{migraphx::shape::float_type, {2, 32, 128}};
    migraphx::module m1;
    {
        auto a   = m1.add_literal(migraphx::generate_literal(as));
        auto b   = m1.add_parameter("input", bs);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), a, b);
        auto lit =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {1, 256, 1}}));
        auto litb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", dot->get_shape().lens()}}), lit);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), dot, litb);
        m1.add_return({mul});
    };
    run_pass(m1);

    migraphx::module m2;
    {
        auto a = m2.add_literal(migraphx::generate_literal(as));
        auto b = m2.add_parameter("input", bs);
        auto lit =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {1, 256, 1}}));
        auto litb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", as.lens()}}), lit);

        auto mul = m2.add_instruction(migraphx::make_op("mul"), a, litb);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), mul, b);
        m2.add_return({dot});
    };

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(dot_mul_b_non_const)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 256, 32}};
    migraphx::shape bs{migraphx::shape::float_type, {2, 32, 128}};
    migraphx::module m1;
    {
        auto a   = m1.add_literal(migraphx::generate_literal(as));
        auto b   = m1.add_parameter("input", bs);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), a, b);
        auto lit =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {1, 1, 128}}));
        auto litb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", dot->get_shape().lens()}}), lit);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), dot, litb);
        m1.add_return({mul});
    };
    migraphx::module m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(dot_slice_b)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 256, 32}};
    migraphx::shape bs{migraphx::shape::float_type, {2, 32, 128}};
    migraphx::module m1;
    {
        auto a     = m1.add_parameter("a", as);
        auto b     = m1.add_parameter("b", bs);
        auto dot   = m1.add_instruction(migraphx::make_op("dot"), a, b);
        auto slice = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {64}}, {"ends", {128}}}), dot);
        m1.add_return({slice});
    };
    run_pass(m1);
    migraphx::module m2;
    {
        auto a       = m2.add_parameter("a", as);
        auto b       = m2.add_parameter("b", bs);
        auto b_slice = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {64}}, {"ends", {128}}}), b);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), a, b_slice);
        m2.add_return({dot});
    };
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(dot_slice_a)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 256, 32}};
    migraphx::shape bs{migraphx::shape::float_type, {2, 32, 128}};
    migraphx::module m1;
    {
        auto a     = m1.add_parameter("a", as);
        auto b     = m1.add_parameter("b", bs);
        auto dot   = m1.add_instruction(migraphx::make_op("dot"), a, b);
        auto slice = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {64}}, {"ends", {128}}}), dot);
        m1.add_return({slice});
    };
    run_pass(m1);
    migraphx::module m2;
    {
        auto a       = m2.add_parameter("a", as);
        auto b       = m2.add_parameter("b", bs);
        auto a_slice = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {64}}, {"ends", {128}}}), a);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), a_slice, b);
        m2.add_return({dot});
    };
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(dot_slice_ab)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 256, 32}};
    migraphx::shape bs{migraphx::shape::float_type, {2, 32, 128}};
    migraphx::module m1;
    {
        auto a     = m1.add_parameter("a", as);
        auto b     = m1.add_parameter("b", bs);
        auto dot   = m1.add_instruction(migraphx::make_op("dot"), a, b);
        auto slice = m1.add_instruction(
            migraphx::make_op("slice",
                              {{"axes", {1, 2}}, {"starts", {64, 64}}, {"ends", {128, 128}}}),
            dot);
        m1.add_return({slice});
    };
    run_pass(m1);
    migraphx::module m2;
    {
        auto a       = m2.add_parameter("a", as);
        auto b       = m2.add_parameter("b", bs);
        auto a_slice = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {64}}, {"ends", {128}}}), a);
        auto b_slice = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {64}}, {"ends", {128}}}), b);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), a_slice, b_slice);
        m2.add_return({dot});
    };
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(dot_slice_batch_dims)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 256, 32}};
    migraphx::shape bs{migraphx::shape::float_type, {2, 32, 128}};
    migraphx::module m1;
    {
        auto a     = m1.add_parameter("a", as);
        auto b     = m1.add_parameter("b", bs);
        auto dot   = m1.add_instruction(migraphx::make_op("dot"), a, b);
        auto slice = m1.add_instruction(
            migraphx::make_op(
                "slice", {{"axes", {0, 1, 2}}, {"starts", {0, 64, 64}}, {"ends", {1, 128, 128}}}),
            dot);
        m1.add_return({slice});
    };
    run_pass(m1);
    migraphx::module m2;
    {
        auto a       = m2.add_parameter("a", as);
        auto b       = m2.add_parameter("b", bs);
        auto a_slice = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {0, 64}}, {"ends", {1, 128}}}),
            a);
        auto b_slice = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0, 2}}, {"starts", {0, 64}}, {"ends", {1, 128}}}),
            b);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), a_slice, b_slice);
        m2.add_return({dot});
    };
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(dot_slice_not_applicable_1)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 256, 32}};
    migraphx::shape bs{migraphx::shape::float_type, {2, 32, 128}};
    migraphx::module m1;
    {
        auto a      = m1.add_parameter("a", as);
        auto b      = m1.add_parameter("b", bs);
        auto dot    = m1.add_instruction(migraphx::make_op("dot"), a, b);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice",
                              {{"axes", {1, 2}}, {"starts", {64, 64}}, {"ends", {128, 128}}}),
            dot);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), dot);

        m1.add_return({slice1, slice2});
    };
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(dot_slice_not_applicable_2)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 256, 32}};
    migraphx::shape bs{migraphx::shape::float_type, {2, 32, 128}};
    migraphx::module m1;
    {
        auto a     = m1.add_parameter("a", as);
        auto b     = m1.add_parameter("b", bs);
        auto dot   = m1.add_instruction(migraphx::make_op("dot"), a, b);
        auto slice = m1.add_instruction(
            migraphx::make_op("slice",
                              {{"axes", {-2, -1}}, {"starts", {64, 64}}, {"ends", {128, 128}}}),
            dot);
        m1.add_return({slice});
    };
    run_pass(m1);
    migraphx::module m2;
    {
        auto a       = m2.add_parameter("a", as);
        auto b       = m2.add_parameter("b", bs);
        auto a_slice = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {64}}, {"ends", {128}}}), a);
        auto b_slice = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {64}}, {"ends", {128}}}), b);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), a_slice, b_slice);
        m2.add_return({dot});
    };
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(conv_concat)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 8, 4, 4}};
    migraphx::shape ws{migraphx::shape::float_type, {2, 8, 3, 3}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", xs);
        auto w      = m1.add_literal(migraphx::generate_literal(ws, 1));
        auto y      = m1.add_parameter("y", xs);
        auto v      = m1.add_literal(migraphx::generate_literal(ws, 2));
        auto conv1  = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        auto conv2  = m1.add_instruction(migraphx::make_op("convolution"), y, v);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), conv1, conv2);
        m1.add_instruction(migraphx::make_op("exp"), concat);
    };
    migraphx::module m2;
    {
        auto x       = m2.add_parameter("x", xs);
        auto w       = m2.add_literal(migraphx::generate_literal(ws, 1));
        auto y       = m2.add_parameter("y", xs);
        auto v       = m2.add_literal(migraphx::generate_literal(ws, 2));
        auto xconcat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
        auto wconcat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), w, v);
        auto conv =
            m2.add_instruction(migraphx::make_op("convolution", {{"group", 2}}), xconcat, wconcat);
        m2.add_instruction(migraphx::make_op("exp"), conv);
    }
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(conv_concat_group)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 8, 4, 4}};
    migraphx::shape ws{migraphx::shape::float_type, {2, 4, 3, 3}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", xs);
        auto w      = m1.add_literal(migraphx::generate_literal(ws, 1));
        auto y      = m1.add_parameter("y", xs);
        auto v      = m1.add_literal(migraphx::generate_literal(ws, 2));
        auto conv1  = m1.add_instruction(migraphx::make_op("convolution", {{"group", 2}}), x, w);
        auto conv2  = m1.add_instruction(migraphx::make_op("convolution", {{"group", 2}}), y, v);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), conv1, conv2);
        m1.add_instruction(migraphx::make_op("exp"), concat);
    };
    migraphx::module m2;
    {
        auto x       = m2.add_parameter("x", xs);
        auto w       = m2.add_literal(migraphx::generate_literal(ws, 1));
        auto y       = m2.add_parameter("y", xs);
        auto v       = m2.add_literal(migraphx::generate_literal(ws, 2));
        auto xconcat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
        auto wconcat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), w, v);
        auto conv =
            m2.add_instruction(migraphx::make_op("convolution", {{"group", 4}}), xconcat, wconcat);
        m2.add_instruction(migraphx::make_op("exp"), conv);
    }
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(find_concat_different_broadcast_axes)
{
    migraphx::shape s1{migraphx::shape::float_type, {128, 1, 1, 1, 1}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 3, 1, 1, 1}};
    migraphx::module m1;
    {
        auto l1  = m1.add_literal(migraphx::generate_literal(s1, 1));
        auto l2  = m1.add_literal(migraphx::generate_literal(s2, 2));
        auto bc1 = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {128, 3, 224, 224, 1}}}), l1);
        auto bc2 = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {128, 3, 224, 224, 1}}}), l2);
        auto cat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 4}}), bc1, bc2);
        m1.add_return({cat});
    };

    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
