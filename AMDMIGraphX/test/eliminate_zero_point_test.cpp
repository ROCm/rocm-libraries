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

#include "migraphx/shape.hpp"
#include <test.hpp>
#include <migraphx/simplify_algebra.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/common.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/apply_alpha_beta.hpp>

static void run_pass(migraphx::module& m) { run_passes(m, {migraphx::simplify_algebra{}}); }

static migraphx::instruction_ref add_quantize_op(migraphx::module& m,
                                                 const std::string& name,
                                                 migraphx::instruction_ref x,
                                                 migraphx::instruction_ref scale,
                                                 migraphx::instruction_ref shift)
{
    auto lens = x->get_shape().lens();
    auto scale_mb =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), scale);
    auto shift_mb =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), shift);
    return m.add_instruction(migraphx::make_op(name), x, scale_mb, shift_mb);
}

static migraphx::instruction_ref
add_quantize_op(migraphx::module& m,
                const std::string& name,
                migraphx::instruction_ref x,
                migraphx::instruction_ref scale,
                migraphx::shape::type_t out_type = migraphx::shape::int8_type)
{
    auto lens = x->get_shape().lens();
    auto scale_mb =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), scale);
    auto op     = migraphx::make_op(name);
    auto op_val = op.to_value();
    if(name == "quantizelinear")
    {
        op_val["out_type"] = to_value(out_type);
    }
    return m.add_instruction(migraphx::make_op(name, op_val), x, scale_mb);
}

static migraphx::instruction_ref add_scale_mul(migraphx::module& m,
                                               migraphx::instruction_ref scale1,
                                               migraphx::instruction_ref scale2,
                                               const std::vector<std::size_t>& out_lens)
{
    auto mul_ins = m.add_instruction(migraphx::make_op("mul"), scale1, scale2);
    if(mul_ins->get_shape().lens() != out_lens)
    {
        mul_ins = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", out_lens}}),
                                    mul_ins);
    }
    return mul_ins;
}

static migraphx::instruction_ref init_zero_point(migraphx::module& m,
                                                 migraphx::instruction_ref q_ins)
{
    auto zp = m.add_literal(migraphx::literal{migraphx::shape{q_ins->get_shape().type()}, {0}});
    return m.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", q_ins->get_shape().lens()}}), zp);
}

TEST_CASE(quantizelinear_ins_no_zp)
{
    migraphx::shape s{migraphx::shape::float_type, {100, 100}};
    migraphx::module m1;
    {
        auto x     = m1.add_parameter("x", s);
        auto scale = m1.add_literal(0.5f);
        auto q_ins = add_quantize_op(m1, "quantizelinear", x, scale);
        m1.add_return({q_ins});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(quantizelinear_ins_with_zp)
{
    migraphx::shape s{migraphx::shape::float_type, {100, 100}};
    migraphx::module m1;
    {
        auto x          = m1.add_parameter("x", s);
        auto scale      = m1.add_literal(0.5f);
        auto zero_point = m1.add_literal(std::int8_t{0});
        auto q_ins      = add_quantize_op(m1, "quantizelinear", x, scale, zero_point);
        m1.add_return({q_ins});
    }
    migraphx::module m2;
    {
        auto x     = m2.add_parameter("x", s);
        auto scale = m2.add_literal(0.5f);
        auto q_ins = add_quantize_op(m2, "quantizelinear", x, scale, migraphx::shape::int8_type);
        m2.add_return({q_ins});
    }
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(quantizelinear_ins_multi_zp_use)
{
    migraphx::shape s{migraphx::shape::float_type, {100, 100}};
    migraphx::module m1;
    {
        auto x     = m1.add_parameter("x", s);
        auto y     = m1.add_parameter("y", migraphx::shape{migraphx::shape::int8_type, s.lens()});
        auto scale = m1.add_literal(0.5f);
        auto zero_point = m1.add_literal(std::int8_t{0});
        auto q_ins      = add_quantize_op(m1, "quantizelinear", x, scale, zero_point);
        auto add_ins    = migraphx::add_common_op(m1, migraphx::make_op("add"), {zero_point, y});
        auto sub_ins    = m1.add_instruction(migraphx::make_op("sub"), {add_ins, q_ins});
        m1.add_return({sub_ins});
    }
    migraphx::module m2;
    {
        auto x       = m2.add_parameter("x", s);
        auto y       = m2.add_parameter("y", migraphx::shape{migraphx::shape::int8_type, s.lens()});
        auto scale   = m2.add_literal(0.5f);
        auto q_ins   = add_quantize_op(m2, "quantizelinear", x, scale, migraphx::shape::int8_type);
        auto sub_ins = m2.add_instruction(migraphx::make_op("sub"), {y, q_ins});
        m2.add_return({sub_ins});
    }
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(dequantizelinear_ins_no_zp)
{
    migraphx::shape s{migraphx::shape::int32_type, {100, 100}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto scale  = m1.add_literal(0.5f);
        auto dq_ins = add_quantize_op(m1, "dequantizelinear", x, scale);
        m1.add_return({dq_ins});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(dequantizelinear_ins_with_zp)
{
    migraphx::shape s{migraphx::shape::int32_type, {100, 100}};
    migraphx::module m1;
    {
        auto x          = m1.add_parameter("x", s);
        auto scale      = m1.add_literal(0.5f);
        auto zero_point = m1.add_literal(std::int32_t{0});
        auto dq_ins     = add_quantize_op(m1, "dequantizelinear", x, scale, zero_point);
        m1.add_return({dq_ins});
    }
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto scale  = m2.add_literal(0.5f);
        auto dq_ins = add_quantize_op(m2, "dequantizelinear", x, scale);
        m2.add_return({dq_ins});
    }
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(dequantizelinear_ins_neg_zero)
{
    migraphx::shape s{migraphx::shape::float_type, {100, 100}};
    migraphx::module m1;
    {
        auto x          = m1.add_parameter("x", s);
        auto scale      = m1.add_literal(migraphx::literal{
            migraphx::shape{migraphx::shape::fp8e4m3fnuz_type, {1}, {0}}, {1.125f}});
        auto zero_point = m1.add_literal(-0.0f);
        auto dq_ins     = add_quantize_op(m1, "dequantizelinear", x, scale, zero_point);
        m1.add_return({dq_ins});
    }
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto scale  = m2.add_literal(migraphx::literal{
            migraphx::shape{migraphx::shape::fp8e4m3fnuz_type, {1}, {0}}, {1.125f}});
        auto dq_ins = add_quantize_op(m2, "dequantizelinear", x, scale);
        m2.add_return({dq_ins});
    }
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(dot)
{
    migraphx::shape sh1{migraphx::shape::float_type, {1280, 1000}};
    migraphx::shape sh2{migraphx::shape::float_type, {1000, 1024}};

    migraphx::module m1;
    {
        auto t1     = m1.add_parameter("t1", sh1);
        auto t2     = m1.add_parameter("t2", sh2);
        auto scale1 = m1.add_literal(0.5f);
        auto scale2 = m1.add_literal(1.5f);
        auto zero1  = m1.add_literal(std::int8_t{0});
        auto zero2  = m1.add_literal(std::int8_t{0});

        auto q1 = add_quantize_op(m1, "quantizelinear", t1, scale1, zero1);
        auto q2 = add_quantize_op(m1, "quantizelinear", t2, scale2, zero2);

        auto dot       = m1.add_instruction(migraphx::make_op("quant_dot"), q1, q2);
        auto out_scale = add_scale_mul(m1, scale1, scale2, dot->get_shape().lens());
        auto out_zp    = init_zero_point(m1, dot);
        auto d3        = add_quantize_op(m1, "dequantizelinear", dot, out_scale, out_zp);
        m1.add_return({d3});
    }
    migraphx::module m2;
    {
        auto t1     = m2.add_parameter("t1", sh1);
        auto t2     = m2.add_parameter("t2", sh2);
        auto scale1 = m2.add_literal(0.5f);
        auto scale2 = m2.add_literal(1.5f);

        auto q1 = add_quantize_op(m2, "quantizelinear", t1, scale1, migraphx::shape::int8_type);
        auto q2 = add_quantize_op(m2, "quantizelinear", t2, scale2, migraphx::shape::int8_type);

        auto dot       = m2.add_instruction(migraphx::make_op("quant_dot"), q1, q2);
        auto out_scale = add_scale_mul(m2, scale1, scale2, dot->get_shape().lens());
        auto d3        = add_quantize_op(m2, "dequantizelinear", dot, out_scale);
        m2.add_return({d3});
    }
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(dot_asymmetric_first_arg)
{
    migraphx::shape sh1{migraphx::shape::float_type, {1280, 1000}};
    migraphx::shape sh2{migraphx::shape::float_type, {1000, 1024}};

    migraphx::module m1;
    {
        auto t1     = m1.add_parameter("t1", sh1);
        auto t2     = m1.add_parameter("t2", sh2);
        auto scale1 = m1.add_literal(0.5f);
        auto scale2 = m1.add_literal(1.5f);

        auto zp1 = m1.add_literal(std::int8_t{1});
        auto zp2 = m1.add_literal(std::int8_t{0});

        auto q1  = add_quantize_op(m1, "quantizelinear", t1, scale1, zp1);
        auto q2  = add_quantize_op(m1, "quantizelinear", t2, scale2, zp2);
        auto dot = m1.add_instruction(migraphx::make_op("quant_dot"), q1, q2);

        auto out_scale = add_scale_mul(m1, scale1, scale2, dot->get_shape().lens());

        auto out_zp = init_zero_point(m1, dot);
        auto zp1_bc = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", t1->get_shape().lens()}}), zp1);

        auto zp_term = m1.add_instruction(migraphx::make_op("quant_dot"), zp1_bc, q2);
        out_zp       = m1.add_instruction(migraphx::make_op("add"), out_zp, zp_term);

        auto d3 = add_quantize_op(m1, "dequantizelinear", dot, out_scale, out_zp);
        m1.add_return({d3});
    }

    run_pass(m1);

    migraphx::module m2;
    {
        auto t1     = m2.add_parameter("t1", sh1);
        auto t2     = m2.add_parameter("t2", sh2);
        auto scale1 = m2.add_literal(0.5f);
        auto scale2 = m2.add_literal(1.5f);
        auto zp1    = m2.add_literal(std::int8_t{1});

        auto q1  = add_quantize_op(m2, "quantizelinear", t1, scale1, zp1);
        auto q2  = add_quantize_op(m2, "quantizelinear", t2, scale2, migraphx::shape::int8_type);
        auto dot = m2.add_instruction(migraphx::make_op("quant_dot"), q1, q2);

        auto out_scale = add_scale_mul(m2, scale1, scale2, dot->get_shape().lens());

        auto zp1_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", t1->get_shape().lens()}}), zp1);

        auto zp_term = m2.add_instruction(migraphx::make_op("quant_dot"), zp1_bc, q2);

        auto d3 = add_quantize_op(m2, "dequantizelinear", dot, out_scale, zp_term);
        m2.add_return({d3});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(dot_asymmetric_both_args)
{
    migraphx::shape sh1{migraphx::shape::float_type, {1280, 1000}};
    migraphx::shape sh2{migraphx::shape::float_type, {1000, 1024}};

    migraphx::module m1;
    {
        auto t1     = m1.add_parameter("t1", sh1);
        auto t2     = m1.add_parameter("t2", sh2);
        auto scale1 = m1.add_literal(0.5f);
        auto scale2 = m1.add_literal(1.5f);

        auto zp1 = m1.add_literal(std::int8_t{2});
        auto zp2 = m1.add_literal(std::int8_t{1});

        auto q1  = add_quantize_op(m1, "quantizelinear", t1, scale1, zp1);
        auto q2  = add_quantize_op(m1, "quantizelinear", t2, scale2, zp2);
        auto dot = m1.add_instruction(migraphx::make_op("quant_dot"), q1, q2);

        auto out_scale = add_scale_mul(m1, scale1, scale2, dot->get_shape().lens());

        auto out_zp = init_zero_point(m1, dot);
        auto zp1_bc = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", t1->get_shape().lens()}}), zp1);
        auto zp2_bc = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", t2->get_shape().lens()}}), zp2);

        auto zp_term1 = m1.add_instruction(migraphx::make_op("quant_dot"), zp1_bc, q2);
        out_zp        = m1.add_instruction(migraphx::make_op("add"), out_zp, zp_term1);
        auto zp_term2 = m1.add_instruction(migraphx::make_op("quant_dot"), q1, zp2_bc);
        out_zp        = m1.add_instruction(migraphx::make_op("add"), out_zp, zp_term2);
        auto zp_term3 = m1.add_instruction(migraphx::make_op("quant_dot"), zp1_bc, zp2_bc);
        out_zp        = m1.add_instruction(migraphx::make_op("sub"), out_zp, zp_term3);

        auto d3 = add_quantize_op(m1, "dequantizelinear", dot, out_scale, out_zp);
        m1.add_return({d3});
    }
    migraphx::module m2;
    {
        auto t1     = m2.add_parameter("t1", sh1);
        auto t2     = m2.add_parameter("t2", sh2);
        auto scale1 = m2.add_literal(0.5f);
        auto scale2 = m2.add_literal(1.5f);

        auto zp1 = m2.add_literal(std::int8_t{2});
        auto zp2 = m2.add_literal(std::int8_t{1});

        auto q1  = add_quantize_op(m2, "quantizelinear", t1, scale1, zp1);
        auto q2  = add_quantize_op(m2, "quantizelinear", t2, scale2, zp2);
        auto dot = m2.add_instruction(migraphx::make_op("quant_dot"), q1, q2);

        auto out_scale = add_scale_mul(m2, scale1, scale2, dot->get_shape().lens());

        auto zp1_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", t1->get_shape().lens()}}), zp1);
        auto zp2_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", t2->get_shape().lens()}}), zp2);

        auto zp_term1 = m2.add_instruction(migraphx::make_op("quant_dot"), zp1_bc, q2);
        auto zp_term2 = m2.add_instruction(migraphx::make_op("quant_dot"), q1, zp2_bc);
        auto out_zp   = m2.add_instruction(migraphx::make_op("add"), zp_term1, zp_term2);
        auto zp_term3 = m2.add_instruction(migraphx::make_op("quant_dot"), zp1_bc, zp2_bc);
        out_zp        = m2.add_instruction(migraphx::make_op("sub"), out_zp, zp_term3);

        auto d3 = add_quantize_op(m2, "dequantizelinear", dot, out_scale, out_zp);
        m2.add_return({d3});
    }
    run_pass(m1);
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
