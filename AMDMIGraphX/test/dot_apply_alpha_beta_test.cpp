/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <cstdint>
#include <migraphx/instruction.hpp>
#include <migraphx/apply_alpha_beta.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>
#include <test.hpp>

TEST_CASE(dot_apply_alpha_beta_half)
{
    migraphx::module m1;
    {
        auto x       = m1.add_parameter("x", migraphx::shape{migraphx::shape::half_type, {2, 2}});
        auto y       = m1.add_parameter("y", migraphx::shape{migraphx::shape::half_type, {2, 2}});
        auto z       = m1.add_parameter("z", migraphx::shape{migraphx::shape::half_type, {2, 2}});
        auto dot_res = migraphx::insert_apply_alpha_beta(
            m1, m1.end(), {x, y, z}, migraphx::make_op("dot"), 3.0f, 2.0f);
        m1.add_instruction(migraphx::make_op("identity"), dot_res);
    }
    migraphx::module m2;
    {

        auto ht              = migraphx::shape::half_type;
        auto ft              = migraphx::shape::float_type;
        auto x               = m2.add_parameter("x", migraphx::shape{ht, {2, 2}});
        auto y               = m2.add_parameter("y", migraphx::shape{ht, {2, 2}});
        auto z               = m2.add_parameter("z", migraphx::shape{ht, {2, 2}});
        auto alpha_literal   = m2.add_literal(3.0f);
        auto alpha_broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", x->get_shape().lens()}}),
            alpha_literal);
        auto x_float = m2.add_instruction(migraphx::make_op("convert", {{"target_type", ft}}), x);
        auto x_alpha_float = m2.add_instruction(migraphx::make_op("mul"), alpha_broadcast, x_float);
        auto x_half =
            m2.add_instruction(migraphx::make_op("convert", {{"target_type", ht}}), x_alpha_float);
        auto dot_res      = m2.add_instruction(migraphx::make_op("dot"), x_half, y);
        auto beta_literal = m2.add_literal(2.0f);
        auto z_float = m2.add_instruction(migraphx::make_op("convert", {{"target_type", ft}}), z);
        auto beta_broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", z->get_shape().lens()}}),
            beta_literal);
        auto z_beta_float = m2.add_instruction(migraphx::make_op("mul"), z_float, beta_broadcast);
        auto z_beta_half =
            m2.add_instruction(migraphx::make_op("convert", {{"target_type", ht}}), z_beta_float);
        auto z_add = m2.add_instruction(migraphx::make_op("add"), dot_res, z_beta_half);
        m2.add_instruction(migraphx::make_op("identity"), z_add);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(dot_apply_alpha_beta_double)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", migraphx::shape{migraphx::shape::double_type, {2, 2}});
        auto y = m1.add_parameter("y", migraphx::shape{migraphx::shape::double_type, {2, 2}});
        auto z = m1.add_parameter("z", migraphx::shape{migraphx::shape::double_type, {2, 1}});
        auto dot_res =
            migraphx::add_apply_alpha_beta(m1, {x, y, z}, migraphx::make_op("dot"), 3.0f, 2.0f);
        m1.add_instruction(migraphx::make_op("identity"), dot_res);
    }
    migraphx::module m2;
    {

        auto dt              = migraphx::shape::double_type;
        auto x               = m2.add_parameter("x", migraphx::shape{dt, {2, 2}});
        auto y               = m2.add_parameter("y", migraphx::shape{dt, {2, 2}});
        auto z               = m2.add_parameter("z", migraphx::shape{dt, {2, 1}});
        auto alpha_literal   = m2.add_literal(3.0f);
        auto alpha_broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", x->get_shape().lens()}}),
            alpha_literal);
        auto alpha_double = m2.add_instruction(migraphx::make_op("convert", {{"target_type", dt}}),
                                               alpha_broadcast);
        auto x_alpha_double = m2.add_instruction(migraphx::make_op("mul"), alpha_double, x);
        auto dot_res        = m2.add_instruction(migraphx::make_op("dot"), x_alpha_double, y);
        auto z_broadcast =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2}}}), z);
        auto beta_literal   = m2.add_literal(2.0f);
        auto beta_broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", z_broadcast->get_shape().lens()}}),
            beta_literal);
        auto beta_double =
            m2.add_instruction(migraphx::make_op("convert", {{"target_type", dt}}), beta_broadcast);
        auto z_beta_double = m2.add_instruction(migraphx::make_op("mul"), z_broadcast, beta_double);
        auto z_add         = m2.add_instruction(migraphx::make_op("add"), dot_res, z_beta_double);
        m2.add_instruction(migraphx::make_op("identity"), z_add);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(quant_dot_apply_alpha_beta)
{
    migraphx::module m1;
    {
        auto x       = m1.add_parameter("x", migraphx::shape{migraphx::shape::int8_type, {2, 2}});
        auto y       = m1.add_parameter("y", migraphx::shape{migraphx::shape::int8_type, {2, 2}});
        auto z       = m1.add_parameter("z", migraphx::shape{migraphx::shape::int32_type, {2, 2}});
        auto dot_res = migraphx::insert_apply_alpha_beta(m1,
                                                         m1.end(),
                                                         {x, y, z},
                                                         migraphx::make_op("quant_dot"),
                                                         migraphx::literal{int32_t{3}},
                                                         migraphx::literal{int32_t{2}});
        m1.add_instruction(migraphx::make_op("identity"), dot_res);
    }
    migraphx::module m2;
    {

        auto i8              = migraphx::shape::int8_type;
        auto i32             = migraphx::shape::int32_type;
        auto x               = m2.add_parameter("x", migraphx::shape{i8, {2, 2}});
        auto y               = m2.add_parameter("y", migraphx::shape{i8, {2, 2}});
        auto z               = m2.add_parameter("z", migraphx::shape{i32, {2, 2}});
        auto alpha_literal   = m2.add_literal(int32_t(3));
        auto alpha_broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", x->get_shape().lens()}}),
            alpha_literal);
        auto x_i32 = m2.add_instruction(migraphx::make_op("convert", {{"target_type", i32}}), x);
        auto x_alpha_i32 = m2.add_instruction(migraphx::make_op("mul"), alpha_broadcast, x_i32);
        auto x_i8 =
            m2.add_instruction(migraphx::make_op("convert", {{"target_type", i8}}), x_alpha_i32);
        auto dot_res        = m2.add_instruction(migraphx::make_op("quant_dot"), x_i8, y);
        auto beta_literal   = m2.add_literal(int32_t(2));
        auto beta_broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", z->get_shape().lens()}}),
            beta_literal);
        auto z_beta_i32 = m2.add_instruction(migraphx::make_op("mul"), z, beta_broadcast);
        auto z_add      = m2.add_instruction(migraphx::make_op("add"), dot_res, z_beta_i32);
        m2.add_instruction(migraphx::make_op("identity"), z_add);
    }
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
