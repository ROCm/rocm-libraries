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

#include <migraphx/literal.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/optimize_module.hpp>
#include <migraphx/propagate_constant.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/serialize.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::optimize_module{}});
}

TEST_CASE(broadcast_transpose_inner_broadcast)
{
    // first optimizes broadcast+transpose to just a broadcast,
    // then finds inner broadcast to become mul+broadcast
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1}, {0}});
        auto y = m1.add_parameter("y", {migraphx::shape::float_type, {1}, {0}});
        auto mb1 =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3}}}), x);
        auto mb2 =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 2}}}), y);
        auto t1 =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), mb1);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), mb2, t1);
        m1.add_return({mul});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x", {migraphx::shape::float_type, {1}, {0}});
        auto y   = m2.add_parameter("y", {migraphx::shape::float_type, {1}, {0}});
        auto mul = m2.add_instruction(migraphx::make_op("mul"), y, x);
        auto mb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 2}}}), mul);
        m2.add_return({mb});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(broadcast_transpose_inner_broadcast_generic)
{
    // first optimizes broadcast+transpose to unsqueeze+transpose+broadcast,
    // then finds inner broadcast to become mul+broadcast
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {5, 10}});
        auto y = m1.add_parameter("y", {migraphx::shape::float_type, {5}});
        auto mb1 =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 5, 10}}}), x);
        auto mb2 =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 10, 5}}}), y);
        auto t1 =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), mb2);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), mb1, t1);
        m1.add_return({mul});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x         = m2.add_parameter("x", {migraphx::shape::float_type, {5, 10}});
        auto y         = m2.add_parameter("y", {migraphx::shape::float_type, {5}});
        auto yb        = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {5, 10}}}), y);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), x, yb);
        auto mb2 = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {3, 5, 10}}}), mul);
        m2.add_return({mb2});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(mul_add_transpose_dot)
{
    auto lit1 = migraphx::generate_literal({migraphx::shape::float_type, {64}}, 0);
    auto lit2 = migraphx::generate_literal({migraphx::shape::float_type, {64}}, 1);
    auto lit3 = migraphx::generate_literal({migraphx::shape::float_type, {64, 64}}, 2);
    migraphx::module m1;
    {
        auto in1      = m1.add_parameter("x", {migraphx::shape::float_type, {2, 64, 4, 4}});
        auto lit1_ins = m1.add_literal(lit1);
        auto lit1_unsq =
            m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 2, 3}}}), lit1_ins);
        auto lit1_mb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 64, 4, 4}}}), lit1_unsq);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), lit1_mb, in1);

        auto lit2_ins = m1.add_literal(lit2);
        auto lit2_unsq =
            m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 2, 3}}}), lit2_ins);
        auto lit2_tp = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), lit2_unsq);
        auto lit2_mb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4, 64}}}), lit2_tp);

        auto mul_tp = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), mul);
        auto add = m1.add_instruction(migraphx::make_op("add"), mul_tp, lit2_mb);

        auto lit3_ins = m1.add_literal(lit3);
        auto lit3_mb  = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 64, 64}}}), lit3_ins);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), add, lit3_mb);

        m1.add_return({dot});
    }
    run_pass(m1);

    // Compute const propagated literals
    migraphx::literal lit13;
    migraphx::literal lit23;
    migraphx::module lit_mod;
    {
        auto lit1_ins  = lit_mod.add_literal(lit1);
        auto lit1_b    = lit_mod.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {64, 64}}}), lit1_ins);

        auto lit3_ins = lit_mod.add_literal(lit3);

        auto mul_lit   = lit_mod.add_instruction(migraphx::make_op("mul"), lit1_b, lit3_ins);
        auto lit13_arg = mul_lit->eval();
        lit13          = migraphx::literal(lit13_arg.get_shape(), lit13_arg.data());

        auto lit2_ins = lit_mod.add_literal(lit2);
        auto lit2_unsq =
            lit_mod.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), lit2_ins);
        auto lit2_mb = lit_mod.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {4, 64}}}), lit2_unsq);

        auto dot_lit   = lit_mod.add_instruction(migraphx::make_op("dot"), lit2_mb, lit3_ins);
        auto lit23_arg = dot_lit->eval();
        lit23          = migraphx::literal(lit23_arg.get_shape(), lit23_arg.data());
    }

    migraphx::module m2;
    {
        auto in1   = m2.add_parameter("x", {migraphx::shape::float_type, {2, 64, 4, 4}});
        auto in_tp = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), in1);

        auto lit13_ins = m2.add_literal(lit13);
        auto lit13_b   = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 64, 64}}}), lit13_ins);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), in_tp, lit13_b);

        auto lit23_ins = m2.add_literal(lit23);
        auto lit23_mb  = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4, 64}}}), lit23_ins);

        auto add = m2.add_instruction(migraphx::make_op("add"), dot, lit23_mb);
        m2.add_return({add});
    }

    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
