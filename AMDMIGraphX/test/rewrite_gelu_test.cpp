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
#include <migraphx/rewrite_gelu.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/reshape.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/ranges.hpp>
#include <test.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/common.hpp>

#include <migraphx/serialize.hpp>

#include <migraphx/verify.hpp>

TEST_CASE(bias_gelu)
{
    migraphx::shape s1{migraphx::shape::half_type, {2, 4, 8}};
    migraphx::shape s2{migraphx::shape::half_type};
    migraphx::module m1;
    {
        auto a    = m1.add_parameter("a", s1);
        auto b    = m1.add_parameter("b", s1);
        auto add1 = m1.add_instruction(migraphx::make_op("add"), a, b);
        auto l1   = m1.add_literal(migraphx::literal{s2, {1.4140625f}});
        auto div  = add_common_op(m1, migraphx::make_op("div"), {add1, l1});
        auto erf  = m1.add_instruction(migraphx::make_op("erf"), div);
        auto l2   = m1.add_literal(migraphx::literal{s2, {1.0f}});
        auto add2 = add_common_op(m1, migraphx::make_op("add"), {erf, l2});
        auto mul  = m1.add_instruction(migraphx::make_op("mul"), add1, add2);
        auto l3   = m1.add_literal(migraphx::literal{s2, {0.5f}});
        mul       = add_common_op(m1, migraphx::make_op("mul"), {mul, l3});
        m1.add_return({mul});
    }
    migraphx::rewrite_gelu pass;
    pass.apply(m1);
    migraphx::dead_code_elimination dce;
    dce.apply(m1);

    migraphx::module m2;
    {
        using migraphx::literal;
        using migraphx::make_op;
        using migraphx::shape;
        auto x_param    = m2.add_parameter("a", s1);
        auto bias_param = m2.add_parameter("b", s1);
        auto bias_add   = m2.add_instruction(migraphx::make_op("add"), x_param, bias_param);
        double const0   = -2. * sqrt(M_2_PI);
        double const1   = 0.044715 * const0;
        auto lit0       = m2.add_literal(literal{shape{s2.type()}, {const0}});
        auto lit1       = m2.add_literal(literal{shape{s2.type()}, {const1}});
        auto one        = m2.add_literal(literal{shape{s2.type()}, {1.0}});
        auto xb         = add_common_op(m2, make_op("mul"), {bias_add, lit1});
        auto a          = m2.add_instruction(make_op("mul"), bias_add, xb);
        auto b          = add_common_op(m2, make_op("add"), {a, lit0});
        auto u          = m2.add_instruction(make_op("mul"), bias_add, b);
        auto emu        = m2.add_instruction(make_op("exp"), u);
        auto c          = add_common_op(m2, make_op("add"), {one, emu});
        auto y          = m2.add_instruction(make_op("div"), bias_add, c);
        m2.add_return({y});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(non_bias_gelu)
{
    migraphx::shape s1{migraphx::shape::half_type, {2, 4, 8}};
    migraphx::shape s2{migraphx::shape::half_type};
    migraphx::module m1;
    {
        auto a    = m1.add_parameter("a", s1);
        auto b    = m1.add_parameter("b", s1);
        auto sub  = m1.add_instruction(migraphx::make_op("sub"), a, b);
        auto l1   = m1.add_literal(migraphx::literal{s2, {1.4140625f}});
        auto div  = add_common_op(m1, migraphx::make_op("div"), {sub, l1});
        auto erf  = m1.add_instruction(migraphx::make_op("erf"), div);
        auto l2   = m1.add_literal(migraphx::literal{s2, {1.0f}});
        auto add2 = add_common_op(m1, migraphx::make_op("add"), {erf, l2});
        auto mul  = m1.add_instruction(migraphx::make_op("mul"), sub, add2);
        auto l3   = m1.add_literal(migraphx::literal{s2, {0.5f}});
        mul       = add_common_op(m1, migraphx::make_op("mul"), {mul, l3});
        m1.add_return({mul});
    }
    migraphx::rewrite_gelu pass;
    pass.apply(m1);
    migraphx::dead_code_elimination dce;
    dce.apply(m1);

    migraphx::module m2;
    {
        using migraphx::literal;
        using migraphx::make_op;
        using migraphx::shape;
        auto x_param    = m2.add_parameter("a", s1);
        auto bias_param = m2.add_parameter("b", s1);
        auto bias_sub   = m2.add_instruction(migraphx::make_op("sub"), x_param, bias_param);
        double const0   = -2. * sqrt(M_2_PI);
        double const1   = 0.044715 * const0;
        auto lit0       = m2.add_literal(literal{shape{s2.type()}, {const0}});
        auto lit1       = m2.add_literal(literal{shape{s2.type()}, {const1}});
        auto one        = m2.add_literal(literal{shape{s2.type()}, {1.0}});
        auto xb         = add_common_op(m2, make_op("mul"), {bias_sub, lit1});
        auto a          = m2.add_instruction(make_op("mul"), bias_sub, xb);
        auto b          = add_common_op(m2, make_op("add"), {a, lit0});
        auto u          = m2.add_instruction(make_op("mul"), bias_sub, b);
        auto emu        = m2.add_instruction(make_op("exp"), u);
        auto c          = add_common_op(m2, make_op("add"), {one, emu});
        auto y          = m2.add_instruction(make_op("div"), bias_sub, c);
        m2.add_return({y});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(tanh_gelu_distilgpt2_fp16)
{
    // Uses constant values seen in the distilgpt2_fp16 model, note how they're not exactly right
    migraphx::shape s1{migraphx::shape::half_type, {2, 4, 8}};
    migraphx::shape s2{migraphx::shape::half_type};
    migraphx::module m1;
    {
        auto x          = m1.add_parameter("x", s1);
        auto fit_const  = m1.add_literal(migraphx::literal{s2, {0.044708251953125}});
        auto sqrt_2_rpi = m1.add_literal(migraphx::literal{s2, {0.7978515625}});
        auto one        = m1.add_literal(migraphx::literal{s2, {1.0f}});
        auto one_half   = m1.add_literal(migraphx::literal{s2, {0.5f}});
        auto three      = m1.add_literal(migraphx::literal{s2, {3.0f}});
        auto pow0       = add_common_op(m1, migraphx::make_op("pow"), {x, three});
        auto mul0       = add_common_op(m1, migraphx::make_op("mul"), {pow0, fit_const});
        auto add0       = m1.add_instruction(migraphx::make_op("add"), {mul0, x});
        auto mul1       = add_common_op(m1, migraphx::make_op("mul"), {add0, sqrt_2_rpi});
        auto tanh0      = m1.add_instruction(migraphx::make_op("tanh"), mul1);
        auto add1       = add_common_op(m1, migraphx::make_op("add"), {tanh0, one});
        auto mul2       = add_common_op(m1, migraphx::make_op("mul"), {x, one_half});
        auto y          = m1.add_instruction(migraphx::make_op("mul"), {add1, mul2});
        m1.add_return({y});
    }
    migraphx::rewrite_gelu pass;
    pass.apply(m1);
    migraphx::dead_code_elimination dce;
    dce.apply(m1);

    migraphx::module m2;
    {
        using migraphx::literal;
        using migraphx::make_op;
        using migraphx::shape;
        auto x        = m2.add_parameter("x", s1);
        double const0 = -2. * sqrt(M_2_PI);
        double const1 = 0.044715 * const0;
        auto lit0     = m2.add_literal(literal{shape{s2.type()}, {const0}});
        auto lit1     = m2.add_literal(literal{shape{s2.type()}, {const1}});
        auto one      = m2.add_literal(literal{shape{s2.type()}, {1.0}});
        auto xb       = add_common_op(m2, make_op("mul"), {x, lit1});
        auto a        = m2.add_instruction(make_op("mul"), x, xb);
        auto b        = add_common_op(m2, make_op("add"), {a, lit0});
        auto u        = m2.add_instruction(make_op("mul"), x, b);
        auto emu      = m2.add_instruction(make_op("exp"), u);
        auto c        = add_common_op(m2, make_op("add"), {one, emu});
        auto y        = m2.add_instruction(make_op("div"), x, c);
        m2.add_return({y});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(fastgelu_fp16)
{
    migraphx::shape s1{migraphx::shape::half_type, {2, 4, 8}};
    migraphx::shape s2{migraphx::shape::half_type};
    migraphx::module m1;
    {
        auto x          = m1.add_parameter("x", s1);
        auto fit_const  = m1.add_literal(migraphx::literal{s2, {0.035677}});
        auto sqrt_2_rpi = m1.add_literal(migraphx::literal{s2, {0.797885}});
        auto one        = m1.add_literal(migraphx::literal{s2, {1.0f}});
        auto one_half   = m1.add_literal(migraphx::literal{s2, {0.5f}});
        auto three      = m1.add_literal(migraphx::literal{s2, {3.0f}});
        auto pow0       = add_common_op(m1, migraphx::make_op("pow"), {x, three});
        auto mul0       = add_common_op(m1, migraphx::make_op("mul"), {pow0, fit_const});
        auto mul1       = add_common_op(m1, migraphx::make_op("mul"), {sqrt_2_rpi, x});
        auto add0       = m1.add_instruction(migraphx::make_op("add"), {mul0, mul1});
        auto tanh0      = m1.add_instruction(migraphx::make_op("tanh"), add0);
        auto add1       = add_common_op(m1, migraphx::make_op("add"), {tanh0, one});
        auto mul2       = add_common_op(m1, migraphx::make_op("mul"), {x, one_half});
        auto y          = m1.add_instruction(migraphx::make_op("mul"), {add1, mul2});
        m1.add_return({y});
    }
    migraphx::rewrite_gelu pass;
    pass.apply(m1);
    migraphx::dead_code_elimination dce;
    dce.apply(m1);

    migraphx::module m2;
    {
        using migraphx::literal;
        using migraphx::make_op;
        using migraphx::shape;
        auto x        = m2.add_parameter("x", s1);
        double const0 = -2. * sqrt(M_2_PI);
        double const1 = 0.044715 * const0;
        auto lit0     = m2.add_literal(literal{shape{s2.type()}, {const0}});
        auto lit1     = m2.add_literal(literal{shape{s2.type()}, {const1}});
        auto one      = m2.add_literal(literal{shape{s2.type()}, {1.0}});
        auto xb       = add_common_op(m2, make_op("mul"), {x, lit1});
        auto a        = m2.add_instruction(make_op("mul"), x, xb);
        auto b        = add_common_op(m2, make_op("add"), {a, lit0});
        auto u        = m2.add_instruction(make_op("mul"), x, b);
        auto emu      = m2.add_instruction(make_op("exp"), u);
        auto c        = add_common_op(m2, make_op("add"), {one, emu});
        auto y        = m2.add_instruction(make_op("div"), x, c);
        m2.add_return({y});
    }

    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
