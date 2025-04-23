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

#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/propagate_constant.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/quantize_int4.hpp>
#include <migraphx/simplify_qdq.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

namespace match = migraphx::match;

TEST_CASE(int4_pass_test)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {16, 8, 6, 6}}));
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        m1.add_instruction(migraphx::make_op("relu"), conv);
    }
    migraphx::run_passes(m1, {migraphx::quantize_int4_pass{}});

    auto chk_1 = match::name("quantizelinear")(
                     match::output(match::name("pack_int4")(match::output(match::name(
                         "unpack_int4")(match::output(match::name("dequantizelinear")))))))
                     .bind("q");

    auto res = find_match(m1, chk_1);

    EXPECT(migraphx::contains(res.instructions, "q"));
}

TEST_CASE(int4_const_prop_test)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {16, 8, 6, 6}}));
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        m1.add_instruction(migraphx::make_op("relu"), conv);
    }
    migraphx::run_passes(m1,
                         {migraphx::quantize_int4_pass{},
                          migraphx::propagate_constant{},
                          migraphx::dead_code_elimination{}});

    auto chk_1 = match::name("pack_int4").bind("pack_int4");
    auto res_1 = find_match(m1, chk_1);
    EXPECT(not migraphx::contains(res_1.instructions, "pack_int4"));

    auto chk_2 = match::name("unpack_int4").bind("unpack_int4");
    auto res_2 = find_match(m1, chk_2);
    EXPECT(migraphx::contains(res_2.instructions, "unpack_int4"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
