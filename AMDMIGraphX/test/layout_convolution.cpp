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
#include <migraphx/layout_convolution.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

static void run_pass(migraphx::module& m, migraphx::layout_convolution lc = {})
{
    migraphx::run_passes(m, {lc, migraphx::dead_code_elimination{}});
}

static migraphx::operation layout(std::vector<int64_t> permutation = {0, 1, 2, 3})
{
    return migraphx::make_op("layout", {{"permutation", permutation}});
}

static migraphx::instruction_ref add_layout_nhwc(migraphx::module& m, migraphx::instruction_ref ins)
{
    return m.add_instruction(layout({0, 2, 3, 1}), ins);
}

static migraphx::instruction_ref add_layout_nchw(migraphx::module& m, migraphx::instruction_ref ins)
{
    return m.add_instruction(layout({0, 1, 2, 3}), ins);
}

TEST_CASE(auto_conv_nchw)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {16, 8, 3, 3}}));
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), conv);
        m1.add_return({relu});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(auto_conv_nhwc)
{
    auto transpose = migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}});
    migraphx::module m1;
    {
        auto x          = m1.add_parameter("x", {migraphx::shape::float_type, {1, 16, 16, 8}});
        auto xtranspose = m1.add_instruction(transpose, x);
        auto w          = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {16, 3, 3, 8}}));
        auto wtranspose = m1.add_instruction(transpose, w);
        auto conv       = m1.add_instruction(
            migraphx::make_op("convolution",
                                    {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            xtranspose,
            wtranspose);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), conv);
        m1.add_return({relu});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x          = m2.add_parameter("x", {migraphx::shape::float_type, {1, 16, 16, 8}});
        auto xtranspose = add_layout_nchw(m2, m2.add_instruction(transpose, x));
        auto w          = m2.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {16, 3, 3, 8}}));
        auto wtranspose = add_layout_nchw(m2, m2.add_instruction(transpose, w));
        auto conv       = m2.add_instruction(
            migraphx::make_op("convolution",
                                    {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            xtranspose,
            wtranspose);
        auto relu = add_layout_nhwc(m2, m2.add_instruction(migraphx::make_op("relu"), conv));
        m2.add_return({relu});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(auto_conv_mixed)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {3, 3, 16, 8}}));
        auto wtranspose =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 3, 0, 1}}}), w);
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            wtranspose);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), conv);
        m1.add_return({relu});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w = m2.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {3, 3, 16, 8}}));
        auto wtranspose =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 3, 0, 1}}}), w);
        auto wlayout = m2.add_instruction(
            migraphx::make_op("layout", {{"permutation", {0, 1, 2, 3}}}), wtranspose);
        auto conv = m2.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            wlayout);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), conv);
        m2.add_return({relu});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(auto_quant_conv_mixed)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::int8_type, {1, 8, 16, 16}});
        auto w =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {3, 3, 16, 8}}));
        auto wtranspose =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 3, 0, 1}}}), w);
        auto conv = m1.add_instruction(
            migraphx::make_op("quant_convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            wtranspose);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), conv);
        m1.add_return({relu});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::int8_type, {1, 8, 16, 16}});
        auto w =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {3, 3, 16, 8}}));
        auto wtranspose =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 3, 0, 1}}}), w);
        auto wlayout = m2.add_instruction(
            migraphx::make_op("layout", {{"permutation", {0, 1, 2, 3}}}), wtranspose);
        auto conv = m2.add_instruction(
            migraphx::make_op("quant_convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            wlayout);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), conv);
        m2.add_return({relu});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_conv_relu)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {16, 8, 3, 3}}));
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        m1.add_instruction(migraphx::make_op("relu"), conv);
    }
    run_pass(m1, {.channels_last = true});

    migraphx::module m2;
    {
        auto x = add_layout_nhwc(
            m2, m2.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}}));
        auto w    = add_layout_nhwc(m2,
                                 m2.add_literal(migraphx::generate_literal(
                                     {migraphx::shape::float_type, {16, 8, 3, 3}})));
        auto conv = m2.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), conv);
        m2.add_instruction(layout(), relu);
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_conv_add)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {16, 8, 3, 3}}));
        auto y    = m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {16}}));
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        auto b = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv->get_shape().lens()}}),
            y);
        m1.add_instruction(migraphx::make_op("add"), conv, b);
    }
    run_pass(m1, {.channels_last = true});

    migraphx::module m2;
    {
        auto x = add_layout_nhwc(
            m2, m2.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}}));
        auto w    = add_layout_nhwc(m2,
                                 m2.add_literal(migraphx::generate_literal(
                                     {migraphx::shape::float_type, {16, 8, 3, 3}})));
        auto y    = m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {16}}));
        auto conv = m2.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        auto b = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv->get_shape().lens()}}),
            y);
        auto add = m2.add_instruction(migraphx::make_op("add"), conv, b);
        m2.add_instruction(layout(), add);
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_quant_conv_add)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::int8_type, {1, 8, 16, 16}});
        auto w =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {16, 8, 3, 3}}));
        auto y    = m1.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {16}}));
        auto conv = m1.add_instruction(
            migraphx::make_op("quant_convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        auto b = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv->get_shape().lens()}}),
            y);
        m1.add_instruction(migraphx::make_op("add"), conv, b);
    }
    run_pass(m1, {.channels_last = true});

    migraphx::module m2;
    {
        auto x = add_layout_nhwc(
            m2, m2.add_parameter("x", {migraphx::shape::int8_type, {1, 8, 16, 16}}));
        auto w    = add_layout_nhwc(m2,
                                 m2.add_literal(migraphx::generate_literal(
                                     {migraphx::shape::int8_type, {16, 8, 3, 3}})));
        auto y    = m2.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {16}}));
        auto conv = m2.add_instruction(
            migraphx::make_op("quant_convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        auto b = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv->get_shape().lens()}}),
            y);
        auto add = m2.add_instruction(migraphx::make_op("add"), conv, b);
        m2.add_instruction(layout(), add);
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_conv_conv)
{
    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", {migraphx::shape::float_type, {1, 2048, 7, 7}});
        auto w1 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {512, 2048, 1, 1}}));
        auto conv1 = m1.add_instruction(migraphx::make_op("convolution"), x, w1);
        auto y1 = m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {512}}));
        auto b1 = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv1->get_shape().lens()}}),
            y1);
        auto add1  = m1.add_instruction(migraphx::make_op("add"), conv1, b1);
        auto relu1 = m1.add_instruction(migraphx::make_op("relu"), add1);
        auto w2    = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {512, 512, 3, 3}}));
        auto conv2 = m1.add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), relu1, w2);
        auto y2 = m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {512}}));
        auto b2 = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv1->get_shape().lens()}}),
            y2);
        auto add2  = m1.add_instruction(migraphx::make_op("add"), conv2, b2);
        auto relu2 = m1.add_instruction(migraphx::make_op("relu"), add2);
        m1.add_return({relu2});
    }
    run_pass(m1, {.channels_last = true});

    migraphx::module m2;
    {
        auto x = add_layout_nhwc(
            m2, m2.add_parameter("x", {migraphx::shape::float_type, {1, 2048, 7, 7}}));
        auto w1    = add_layout_nhwc(m2,
                                  m2.add_literal(migraphx::generate_literal(
                                      {migraphx::shape::float_type, {512, 2048, 1, 1}})));
        auto conv1 = m2.add_instruction(migraphx::make_op("convolution"), x, w1);
        auto y1 = m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {512}}));
        auto b1 = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv1->get_shape().lens()}}),
            y1);
        auto add1  = m2.add_instruction(migraphx::make_op("add"), conv1, b1);
        auto relu1 = m2.add_instruction(migraphx::make_op("relu"), add1);
        auto w2    = add_layout_nhwc(m2,
                                  m2.add_literal(migraphx::generate_literal(
                                      {migraphx::shape::float_type, {512, 512, 3, 3}})));
        auto conv2 = m2.add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), relu1, w2);
        auto y2 = m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {512}}));
        auto b2 = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv1->get_shape().lens()}}),
            y2);
        auto add2        = m2.add_instruction(migraphx::make_op("add"), conv2, b2);
        auto relu2       = m2.add_instruction(migraphx::make_op("relu"), add2);
        auto relu_layout = m2.add_instruction(layout(), relu2);
        m2.add_return({relu_layout});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_conv_reduce)
{
    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", {migraphx::shape::float_type, {1, 2048, 7, 7}});
        auto w1 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {512, 2048, 1, 1}}));
        auto conv1 = m1.add_instruction(migraphx::make_op("convolution"), x, w1);
        auto y1 = m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {512}}));
        auto b1 = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv1->get_shape().lens()}}),
            y1);
        auto add1  = m1.add_instruction(migraphx::make_op("add"), conv1, b1);
        auto relu1 = m1.add_instruction(migraphx::make_op("relu"), add1);
        auto reduce =
            m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2, 3}}}), relu1);
        auto squeeze = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {2, 3}}}), reduce);
        m1.add_return({squeeze});
    }
    run_pass(m1, {.channels_last = true});

    migraphx::module m2;
    {
        auto x = add_layout_nhwc(
            m2, m2.add_parameter("x", {migraphx::shape::float_type, {1, 2048, 7, 7}}));
        auto w1    = add_layout_nhwc(m2,
                                  m2.add_literal(migraphx::generate_literal(
                                      {migraphx::shape::float_type, {512, 2048, 1, 1}})));
        auto conv1 = m2.add_instruction(migraphx::make_op("convolution"), x, w1);
        auto y1 = m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {512}}));
        auto b1 = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv1->get_shape().lens()}}),
            y1);
        auto add1  = m2.add_instruction(migraphx::make_op("add"), conv1, b1);
        auto relu1 = m2.add_instruction(migraphx::make_op("relu"), add1);
        auto reduce =
            m2.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2, 3}}}), relu1);
        auto squeeze = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {2, 3}}}), reduce);
        m2.add_return({squeeze});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_group_conv)
{
    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", {migraphx::shape::float_type, {64, 96, 80, 80}});
        auto w1 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {64, 96, 1, 1}}));
        auto w2 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {64, 1, 3, 3}}));
        auto conv1 = m1.add_instruction(migraphx::make_op("convolution"), x, w1);
        auto relu1 = m1.add_instruction(migraphx::make_op("relu"), conv1);
        auto conv2 =
            m1.add_instruction(migraphx::make_op("convolution", {{"group", 64}}), relu1, w2);
        auto relu2 = m1.add_instruction(migraphx::make_op("relu"), conv2);
        m1.add_return({relu2});
    }
    run_pass(m1, {.channels_last = true});

    migraphx::module m2;
    {
        auto x = add_layout_nhwc(
            m2, m2.add_parameter("x", {migraphx::shape::float_type, {64, 96, 80, 80}}));
        auto w1    = add_layout_nhwc(m2,
                                  m2.add_literal(migraphx::generate_literal(
                                      {migraphx::shape::float_type, {64, 96, 1, 1}})));
        auto w2    = add_layout_nchw(m2,
                                  m2.add_literal(migraphx::generate_literal(
                                      {migraphx::shape::float_type, {64, 1, 3, 3}})));
        auto conv1 = m2.add_instruction(migraphx::make_op("convolution"), x, w1);
        auto relu1 = add_layout_nhwc(m2, m2.add_instruction(migraphx::make_op("relu"), conv1));
        auto conv2 =
            m2.add_instruction(migraphx::make_op("convolution", {{"group", 64}}), relu1, w2);
        auto relu2 = add_layout_nchw(m2, m2.add_instruction(migraphx::make_op("relu"), conv2));
        m2.add_return({relu2});
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
