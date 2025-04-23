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
#include <cmath>
#include <limits>

#include <migraphx/float8.hpp>
#include <migraphx/half.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/fp8_ocp_to_fnuz.hpp>
#include <migraphx/simplify_qdq.hpp>
#include <migraphx/propagate_constant.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>
#include <quantize_helpers.hpp>

using migraphx::make_op;
using migraphx::shape;
using migraphx::fp8::fp8e4m3fnuz;

static void run_fp8_ocp_to_fnuz(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::fp8_ocp_to_fnuz{}, migraphx::dead_code_elimination{}});
}

static void run_simplify_qdq(migraphx::module& m)
{
    run_passes(m, {migraphx::simplify_qdq{}, migraphx::dead_code_elimination{}});
}

static void run_cse_pc(migraphx::module& m, const std::unordered_set<std::string>& skip_ops = {})
{
    run_passes(m,
               {migraphx::eliminate_common_subexpression{},
                migraphx::dead_code_elimination{},
                migraphx::propagate_constant{skip_ops},
                migraphx::dead_code_elimination{}});
}

static auto bit_cast_and_handle_specials(migraphx::module& m,
                                         const migraphx::instruction_ref x,
                                         const migraphx::instruction_ref bits_0x80_lit,
                                         const migraphx::instruction_ref bits_0x7f_lit,
                                         const migraphx::instruction_ref bits_0xff_lit,
                                         const migraphx::instruction_ref bits_0x00_lit)
{
    auto x_lens = x->get_shape().lens();
    auto cast_input =
        m.add_instruction(make_op("bit_cast", {{"target_type", shape::fp8e4m3fnuz_type}}), x);
    auto mb_bits_0x80_lit =
        m.add_instruction(make_op("multibroadcast", {{"out_lens", x_lens}}), bits_0x80_lit);
    auto mb_bits_0x7f_lit =
        m.add_instruction(make_op("multibroadcast", {{"out_lens", x_lens}}), bits_0x7f_lit);
    auto mb_bits_0xff_lit =
        m.add_instruction(make_op("multibroadcast", {{"out_lens", x_lens}}), bits_0xff_lit);
    auto mb_zero_lit =
        m.add_instruction(make_op("multibroadcast", {{"out_lens", x_lens}}), bits_0x00_lit);
    // negative zero in fp8e4m3fn to zero in fp8e4m3fnuz
    // a == 0x80 ? 0x0 : a
    auto is_neg_zero = m.add_instruction(make_op("equal"), cast_input, mb_bits_0x80_lit);
    auto ret         = m.add_instruction(make_op("where"), is_neg_zero, mb_zero_lit, cast_input);

    // positive and negative NaN in fp8e4m3fn to NaN in fp8e4m3fnuz
    // (a == 0x7f or a == 0xff) ? 0x80 : a
    auto eq_0x7f = m.add_instruction(make_op("equal"), ret, mb_bits_0x7f_lit);

    auto eq_0xff = m.add_instruction(make_op("equal"), ret, mb_bits_0xff_lit);

    auto cond = m.add_instruction(make_op("logical_or"), eq_0x7f, eq_0xff);
    ret       = m.add_instruction(make_op("where"), cond, mb_bits_0x80_lit, ret);
    return ret;
}

static auto cast_fp8_helper(migraphx::module& m,
                            const migraphx::instruction_ref dq_input,
                            const migraphx::instruction_ref dq_scale,
                            const migraphx::instruction_ref dq_zp)
{
    auto dq_input_lens                 = dq_input->get_shape().lens();
    std::vector<fp8e4m3fnuz> bits_0x80 = {fp8e4m3fnuz(0x80, fp8e4m3fnuz::from_bits())};
    std::vector<fp8e4m3fnuz> bits_0x7f = {fp8e4m3fnuz(0x7f, fp8e4m3fnuz::from_bits())};
    std::vector<fp8e4m3fnuz> bits_0xff = {fp8e4m3fnuz(0xff, fp8e4m3fnuz::from_bits())};
    std::vector<fp8e4m3fnuz> bits_0x00 = {fp8e4m3fnuz(0x00, fp8e4m3fnuz::from_bits())};
    auto bits_0x80_lit = m.add_literal(shape{shape::fp8e4m3fnuz_type, {1}, {0}}, bits_0x80);
    auto bits_0x7f_lit = m.add_literal(shape{shape::fp8e4m3fnuz_type, {1}, {0}}, bits_0x7f);
    auto bits_0xff_lit = m.add_literal(shape{shape::fp8e4m3fnuz_type, {1}, {0}}, bits_0xff);
    auto bits_0x00_lit = m.add_literal(shape{shape::fp8e4m3fnuz_type, {1}, {0}}, bits_0x00);

    auto cast_input = bit_cast_and_handle_specials(
        m, dq_input, bits_0x80_lit, bits_0x7f_lit, bits_0xff_lit, bits_0x00_lit);
    auto adj_zp = bit_cast_and_handle_specials(
        m, dq_zp, bits_0x80_lit, bits_0x7f_lit, bits_0xff_lit, bits_0x00_lit);

    auto two_lit = m.add_literal(migraphx::literal{shape{dq_scale->get_shape().type()}, {2}});
    two_lit      = m.add_instruction(
        make_op("multibroadcast", {{"out_lens", dq_scale->get_shape().lens()}}), two_lit);
    auto adj_dq_scale = m.add_instruction(make_op("mul"), dq_scale, two_lit);

    return std::vector<migraphx::instruction_ref>{cast_input, adj_dq_scale, adj_zp};
}

TEST_CASE(fp8_gemm_conversion)
{
    using migraphx::fp8::fp8e4m3fn;
    using migraphx::fp8::fp8e4m3fnuz;
    std::vector<std::size_t> data_lens = {2, 3, 8, 8};
    migraphx::module m1;
    {
        auto a     = m1.add_parameter("a", {migraphx::shape::float_type, data_lens});
        auto b     = m1.add_parameter("b", {migraphx::shape::float_type, data_lens});
        auto scale = m1.add_literal(0.5f);
        std::vector<fp8e4m3fn> data;
        data.push_back(fp8e4m3fn{0.f});
        auto zero =
            m1.add_literal(migraphx::shape{migraphx::shape::fp8e4m3fn_type, {1}, {0}}, data);

        auto qa = add_quantize_op(m1, "quantizelinear", a, scale, zero);
        auto qb = add_quantize_op(m1, "quantizelinear", b, scale, zero);
        auto da =
            add_quantize_op(m1, "dequantizelinear", qa, qa->inputs().at(1), qa->inputs().at(2));
        auto db =
            add_quantize_op(m1, "dequantizelinear", qb, qb->inputs().at(1), qb->inputs().at(2));
        auto dot = m1.add_instruction(migraphx::make_op("dot"), da, db);
        m1.add_return({dot});
    }
    run_fp8_ocp_to_fnuz(m1);

    // expected after fp8_ocp_to_fnuz
    migraphx::module m2;
    {
        auto a     = m2.add_parameter("a", {migraphx::shape::float_type, data_lens});
        auto b     = m2.add_parameter("b", {migraphx::shape::float_type, data_lens});
        auto scale = m2.add_literal(0.5f);
        std::vector<fp8e4m3fn> data;
        data.push_back(fp8e4m3fn{0.f});
        auto zero =
            m2.add_literal(migraphx::shape{migraphx::shape::fp8e4m3fn_type, {1}, {0}}, data);

        auto qa = add_quantize_op(m2, "quantizelinear", a, scale, zero);
        auto qb = add_quantize_op(m2, "quantizelinear", b, scale, zero);

        auto outs_a = cast_fp8_helper(m2, qa, scale, zero);
        auto adj_a  = outs_a.at(0);
        auto mb_scales_a =
            m2.add_instruction(make_op("multibroadcast", {{"out_lens", data_lens}}), outs_a.at(1));
        auto mb_zp_a =
            m2.add_instruction(make_op("multibroadcast", {{"out_lens", data_lens}}), outs_a.at(2));
        auto da = m2.add_instruction(make_op("dequantizelinear"), adj_a, mb_scales_a, mb_zp_a);

        auto outs_b = cast_fp8_helper(m2, qb, scale, zero);
        auto adj_b  = outs_b.at(0);
        auto mb_scales_b =
            m2.add_instruction(make_op("multibroadcast", {{"out_lens", data_lens}}), outs_b.at(1));
        auto mb_zp_b =
            m2.add_instruction(make_op("multibroadcast", {{"out_lens", data_lens}}), outs_b.at(2));
        auto db = m2.add_instruction(make_op("dequantizelinear"), adj_b, mb_scales_b, mb_zp_b);

        auto dot = m2.add_instruction(migraphx::make_op("dot"), da, db);
        m2.add_return({dot});
    }

    EXPECT(m1 == m2);

    // expected after simplify_qdq
    migraphx::module m3;
    {
        auto a     = m3.add_parameter("a", {migraphx::shape::float_type, {2, 3, 8, 8}});
        auto b     = m3.add_parameter("b", {migraphx::shape::float_type, {2, 3, 8, 8}});
        auto scale = m3.add_literal(0.5f);
        std::vector<fp8e4m3fn> data;
        data.push_back(fp8e4m3fn{0.f});
        auto zero =
            m3.add_literal(migraphx::shape{migraphx::shape::fp8e4m3fn_type, {1}, {0}}, data);

        auto qa = add_quantize_op(m3, "quantizelinear", a, scale, zero);
        auto qb = add_quantize_op(m3, "quantizelinear", b, scale, zero);

        auto outs_a      = cast_fp8_helper(m3, qa, qa->inputs().at(1), qa->inputs().at(2));
        auto outs_b      = cast_fp8_helper(m3, qb, qb->inputs().at(1), qb->inputs().at(2));
        auto adj_qa      = outs_a.at(0);
        auto adj_scale_a = outs_a.at(1);
        auto adj_qb      = outs_b.at(0);
        auto adj_scale_b = outs_b.at(1);

        auto dot = m3.add_instruction(migraphx::make_op("quant_dot"), adj_qa, adj_qb);

        auto out_scale = add_scale_mul(m3, adj_scale_a, adj_scale_b, 1, 1, dot->get_shape().lens());
        auto dq_out    = add_quantize_op(m3, "dequantizelinear", dot, out_scale);
        m3.add_return({dq_out});
    }

    run_simplify_qdq(m1);
    // running propagate constant to simplify adjustments to literals
    // could pass the test without, but a tedious amount of instructions to rearrange
    run_cse_pc(m1);
    run_cse_pc(m3);
    EXPECT(m1 == m3);
    m1.debug_print();
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
