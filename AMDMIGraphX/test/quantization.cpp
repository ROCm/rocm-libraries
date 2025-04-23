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
#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/apply_alpha_beta.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/truncate_float.hpp>
#include <migraphx/quantize_8bits.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/propagate_constant.hpp>
#include <migraphx/simplify_qdq.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/program.hpp>
#include <migraphx/shape.hpp>
#include "test.hpp"
#include <migraphx/half.hpp>

static void optimize_prog_int8(migraphx::program& prog)
{
    migraphx::run_passes(prog,
                         {migraphx::simplify_qdq{},
                          migraphx::eliminate_common_subexpression{},
                          migraphx::dead_code_elimination{}});
}

TEST_CASE(param_add)
{
    auto create_program_float = [](bool add_return = false) {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1  = mm->add_parameter("x", s);
        auto p2  = mm->add_parameter("y", s);
        auto sum = mm->add_instruction(migraphx::make_op("add"), p1, p2);
        if(add_return)
        {
            mm->add_return({sum});
        }

        return p;
    };

    auto create_program_half = [](bool add_return = false) {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1  = mm->add_parameter("x", s);
        auto p2  = mm->add_parameter("y", s);
        auto hp1 = mm->add_instruction(migraphx::make_op("convert"), p1);
        auto hp2 = mm->add_instruction(migraphx::make_op("convert"), p2);
        auto hs  = mm->add_instruction(migraphx::make_op("add"), hp1, hp2);
        auto fs  = mm->add_instruction(
            migraphx::make_op("convert",
                               {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            hs);
        if(add_return)
        {
            mm->add_return({fs});
        }
        else
        {
            mm->add_instruction(migraphx::make_op("identity"), {fs});
        }

        return p;
    };

    {
        auto p1 = create_program_float();
        auto p2 = create_program_half();

        migraphx::quantize_fp16(p1);
        EXPECT(p1 == p2);
    }

    {
        auto p1 = create_program_float();
        auto p2 = create_program_half();

        migraphx::quantize_fp16(p1, {"add"});
        EXPECT(p1 == p2);
    }

    {
        auto p1 = create_program_float(true);
        auto p2 = create_program_half(true);

        migraphx::quantize_fp16(p1);
        EXPECT(p1 == p2);
    }

    {
        auto p1 = create_program_float(true);
        auto p2 = create_program_half(true);

        migraphx::quantize_fp16(p1, {"add"});
        EXPECT(p1 == p2);
    }
}

TEST_CASE(param_add_sub)
{
    auto create_program_float = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1   = mm->add_parameter("x", s);
        auto p2   = mm->add_parameter("y", s);
        auto sum  = mm->add_instruction(migraphx::make_op("add"), p1, p2);
        auto diff = mm->add_instruction(migraphx::make_op("sub"), sum, p2);
        auto r    = mm->add_instruction(migraphx::make_op("add"), diff, p1);
        mm->add_return({r});

        return p;
    };

    auto create_program_half_add = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1  = mm->add_parameter("x", s);
        auto p2  = mm->add_parameter("y", s);
        auto hp1 = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), p1);
        auto hp2 = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), p2);
        auto hsum = mm->add_instruction(migraphx::make_op("add"), hp1, hp2);
        auto sum  = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), hsum);
        auto diff  = mm->add_instruction(migraphx::make_op("sub"), sum, p2);
        auto hdiff = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), diff);
        auto hadd = mm->add_instruction(migraphx::make_op("add"), hdiff, hp1);
        auto fadd = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), hadd);
        mm->add_return({fadd});

        return p;
    };

    auto create_program_half_sub = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1   = mm->add_parameter("x", s);
        auto p2   = mm->add_parameter("y", s);
        auto sum  = mm->add_instruction(migraphx::make_op("add"), p1, p2);
        auto hsum = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), sum);
        auto hp2 = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), p2);
        auto hdiff = mm->add_instruction(migraphx::make_op("sub"), hsum, hp2);
        auto diff  = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), hdiff);
        auto r = mm->add_instruction(migraphx::make_op("add"), diff, p1);
        mm->add_return({r});

        return p;
    };

    {
        auto p1 = create_program_float();
        auto p2 = create_program_half_add();

        migraphx::quantize_fp16(p1, {"add"});
        EXPECT(p1 == p2);
    }

    {
        auto p1 = create_program_float();
        auto p2 = create_program_half_sub();

        migraphx::quantize_fp16(p1, {"sub"});

        EXPECT(p1 == p2);
    }

    {
        auto create_program_fp16 = [] {
            migraphx::program p;
            auto* mm = p.get_main_module();
            migraphx::shape s{migraphx::shape::float_type, {2, 3}};
            auto p1  = mm->add_parameter("x", s);
            auto p2  = mm->add_parameter("y", s);
            auto hp1 = mm->add_instruction(
                migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), p1);
            auto hp2 = mm->add_instruction(
                migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), p2);
            auto hsum = mm->add_instruction(migraphx::make_op("add"), hp1, hp2);
            auto sum  = mm->add_instruction(
                migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), hsum);
            auto hsum1 = mm->add_instruction(
                migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), sum);
            auto p3 = mm->add_instruction(
                migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), p2);
            auto diff  = mm->add_instruction(migraphx::make_op("sub"), hsum1, p3);
            auto fdiff = mm->add_instruction(
                migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), diff);
            auto hdiff1 = mm->add_instruction(
                migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), fdiff);
            auto p4 = mm->add_instruction(
                migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), p1);
            auto res = mm->add_instruction(migraphx::make_op("add"), hdiff1, p4);
            auto r   = mm->add_instruction(
                migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), res);
            mm->add_return({r});

            return p;
        };

        auto create_program_quant_fp16 = [] {
            migraphx::program p;
            auto* mm = p.get_main_module();
            migraphx::shape s{migraphx::shape::float_type, {2, 3}};
            auto p1  = mm->add_parameter("x", s);
            auto p2  = mm->add_parameter("y", s);
            auto hp1 = mm->add_instruction(
                migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), p1);
            auto hp2 = mm->add_instruction(
                migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), p2);
            auto hsum  = mm->add_instruction(migraphx::make_op("add"), hp1, hp2);
            auto hdiff = mm->add_instruction(migraphx::make_op("sub"), hsum, hp2);
            auto hres  = mm->add_instruction(migraphx::make_op("add"), hdiff, hp1);
            auto r     = mm->add_instruction(
                migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), hres);
            mm->add_return({r});

            return p;
        };

        auto p0 = create_program_float();
        migraphx::run_passes(p0,
                             {migraphx::truncate_float_pass{{"all"}, migraphx::shape::half_type},
                              migraphx::dead_code_elimination{}});
        EXPECT(p0 == create_program_fp16());

        auto p1 = create_program_float();
        migraphx::quantize_fp16(p1);
        EXPECT(p1 == create_program_quant_fp16());
    }
}

TEST_CASE(literal_add)
{
    auto create_program_float = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        std::vector<float> data(2 * 3);
        std::iota(data.begin(), data.end(), 1.0f);
        auto l1 = mm->add_literal(migraphx::literal(s, data));
        auto l2 = mm->add_literal(migraphx::literal(s, data));
        mm->add_instruction(migraphx::make_op("add"), l1, l2);
        return p;
    };

    auto create_program_half = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::half_type, {2, 3}};
        std::vector<migraphx::half> data(2 * 3);
        std::iota(data.begin(), data.end(), 1.0f);
        auto l1 = mm->add_literal(migraphx::literal(s, data));
        auto l2 = mm->add_literal(migraphx::literal(s, data));
        auto hs = mm->add_instruction(migraphx::make_op("add"), l1, l2);
        auto fs = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            hs);
        mm->add_instruction(migraphx::make_op("identity"), fs);
        return p;
    };

    {
        auto p1 = create_program_float();
        auto p2 = create_program_half();

        migraphx::quantize_fp16(p1, {"all"});
        migraphx::run_passes(*p1.get_main_module(),
                             {migraphx::propagate_constant{}, migraphx::dead_code_elimination{}});
        migraphx::run_passes(*p2.get_main_module(),
                             {migraphx::propagate_constant{}, migraphx::dead_code_elimination{}});

        EXPECT(p1 == p2);
    }

    {
        auto p1 = create_program_float();
        auto p2 = create_program_half();

        migraphx::quantize_fp16(p1, {"add"});
        migraphx::run_passes(*p1.get_main_module(),
                             {migraphx::propagate_constant{}, migraphx::dead_code_elimination{}});
        migraphx::run_passes(*p2.get_main_module(),
                             {migraphx::propagate_constant{}, migraphx::dead_code_elimination{}});
        EXPECT(p1 == p2);
    }
}

TEST_CASE(fp16_subgraph)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sd{migraphx::shape::float_type, {1}};
        auto l1 = mm->add_literal(migraphx::literal(sd, {1}));
        auto l2 = mm->add_literal(migraphx::literal(sd, {2}));
        auto l3 = mm->add_literal(migraphx::literal(sd, {3}));
        migraphx::shape sx{migraphx::shape::float_type, {1, 4}};
        migraphx::shape sy{migraphx::shape::float_type, {3, 4}};
        migraphx::shape sc{migraphx::shape::bool_type};
        auto cond = mm->add_parameter("cond", sc);
        auto x    = mm->add_parameter("x", sx);
        auto y    = mm->add_parameter("y", sy);

        auto* then_mod = p.create_module("If_6_if");
        auto m1        = then_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 4}}}), l1);
        auto add0 = then_mod->add_instruction(migraphx::make_op("add"), x, m1);
        auto m2   = then_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {3, 4}}}), l2);
        auto mul0  = then_mod->add_instruction(migraphx::make_op("mul"), y, m2);
        auto mfp16 = then_mod->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), mul0);
        then_mod->add_return({add0, mul0, mfp16});

        auto* else_mod = p.create_module("If_6_else");
        auto me1       = else_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 4}}}), l3);
        auto mul1 = else_mod->add_instruction(migraphx::make_op("mul"), x, me1);
        auto me2  = else_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {3, 4}}}), l3);
        auto add1  = else_mod->add_instruction(migraphx::make_op("add"), y, me2);
        auto afp16 = else_mod->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), add1);
        else_mod->add_return({mul1, add1, afp16});

        auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        auto r0  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
        auto r1  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), ret);
        auto r16 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), ret);
        mm->add_return({r0, r1, r16});

        return p;
    };

    auto create_fp16_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sd{migraphx::shape::half_type, {1}};
        migraphx::shape sx{migraphx::shape::float_type, {1, 4}};
        migraphx::shape sy{migraphx::shape::float_type, {3, 4}};
        migraphx::shape sc{migraphx::shape::bool_type};
        auto cond      = mm->add_parameter("cond", sc);
        auto x         = mm->add_parameter("x", sx);
        auto y         = mm->add_parameter("y", sy);
        auto* then_mod = p.create_module("If_6_if");
        auto hl2       = then_mod->add_literal(migraphx::literal(sd, {2}));
        auto hl1       = then_mod->add_literal(migraphx::literal(sd, {1}));
        auto mhl1      = then_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 4}}}), hl1);
        auto hx = then_mod->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), x);
        auto ad  = then_mod->add_instruction(migraphx::make_op("add"), hx, mhl1);
        auto fad = then_mod->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), ad);
        auto mhl2 = then_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {3, 4}}}), hl2);
        auto hy1 = then_mod->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), y);
        auto mu  = then_mod->add_instruction(migraphx::make_op("mul"), hy1, mhl2);
        auto fmu = then_mod->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), mu);
        then_mod->add_return({fad, fmu, mu});

        auto* else_mod = p.create_module("If_6_else");
        auto hl3       = else_mod->add_literal(migraphx::literal(sd, {3}));
        auto mhl3      = else_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 4}}}), hl3);
        auto hx2 = else_mod->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), x);
        auto mu1  = else_mod->add_instruction(migraphx::make_op("mul"), hx2, mhl3);
        auto fmu1 = else_mod->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), mu1);
        auto mhl4 = else_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {3, 4}}}), hl3);
        auto hy = else_mod->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), y);
        auto ad1  = else_mod->add_instruction(migraphx::make_op("add"), hy, mhl4);
        auto fad1 = else_mod->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), ad1);
        else_mod->add_return({fmu1, fad1, ad1});

        auto iff = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        auto r0  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), iff);
        auto r1  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), iff);
        auto r2  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), iff);
        mm->add_return({r0, r1, r2});

        return p;
    };

    auto p1 = create_program();
    migraphx::quantize_fp16(p1);

    auto p2 = create_fp16_program();

    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(topk)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3, 5}};
        auto data = mm->add_parameter("data", s);
        auto r    = mm->add_instruction(
            migraphx::make_op("topk", {{"axis", 0}, {"k", 3}, {"largest", 0}}), data);
        auto r0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), r);
        auto r1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), r);
        mm->add_return({r0, r1});

        return p;
    };

    auto create_fp16_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3, 5}};
        auto fdata = mm->add_parameter("data", s);
        auto hdata = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), fdata);
        auto r = mm->add_instruction(
            migraphx::make_op("topk", {{"axis", 0}, {"k", 3}, {"largest", 0}}), hdata);
        auto hr0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), r);
        auto fr0 = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), hr0);
        auto ir1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), r);
        mm->add_return({fr0, ir1});

        return p;
    };

    auto p1 = create_program();
    migraphx::quantize_fp16(p1);

    auto p2 = create_fp16_program();

    EXPECT(p1 == p2);
}

TEST_CASE(op_capture)
{
    auto create_program_float = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s1{migraphx::shape::float_type, {3, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {3, 6}};

        auto p1 = mm->add_parameter("x", s1);
        auto p2 = mm->add_parameter("y", s1);
        auto pb = mm->add_parameter("b", s2);
        auto pc = mm->add_parameter("c", s2);
        auto pa = mm->add_instruction(migraphx::make_op("add"), p1, p2);
        auto ps =
            migraphx::add_apply_alpha_beta(*mm, {pa, pb, pc}, migraphx::make_op("dot"), 1.0f, 1.0f);
        mm->add_instruction(migraphx::make_op("dot"), pa, ps);

        return p;
    };

    auto create_program_op = [&] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s1{migraphx::shape::float_type, {3, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {3, 6}};

        auto p1  = mm->add_parameter("x", s1);
        auto p2  = mm->add_parameter("y", s1);
        auto pb  = mm->add_parameter("b", s2);
        auto pc  = mm->add_parameter("c", s2);
        auto pa  = mm->add_instruction(migraphx::make_op("add"), p1, p2);
        auto opa = mm->add_instruction(migraphx::make_op("capture", {{"ins_index", 0}}), pa);
        auto opb = mm->add_instruction(migraphx::make_op("capture", {{"ins_index", 1}}), pb);
        auto ps  = migraphx::add_apply_alpha_beta(
            *mm, {opa, opb, pc}, migraphx::make_op("dot"), 1.0f, 1.0f);
        auto opm = mm->add_instruction(migraphx::make_op("capture", {{"ins_index", 2}}), pa);
        auto ops = mm->add_instruction(migraphx::make_op("capture", {{"ins_index", 3}}), ps);
        mm->add_instruction(migraphx::make_op("dot"), opm, ops);

        return p;
    };

    {
        auto p                  = create_program_float();
        auto op_capture_p       = create_program_op();
        migraphx::target t      = migraphx::make_target("ref");
        std::size_t param_index = 0;
        migraphx::run_passes(
            p, {migraphx::capture_arguments_pass{{"dot", "convolution"}, {}, &param_index}});
        EXPECT(p == op_capture_p);
    }
}

TEST_CASE(op_capture_subgraph)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sx{migraphx::shape::float_type, {2, 2, 4, 8}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 2, 8, 6}};
        migraphx::shape sc{migraphx::shape::bool_type};
        auto cond = mm->add_parameter("cond", sc);
        auto a    = mm->add_parameter("a", sx);
        auto b    = mm->add_parameter("b", sy);

        migraphx::shape sd{migraphx::shape::float_type, {2, 2, 4, 6}};
        migraphx::shape sw{migraphx::shape::float_type, {2, 2, 1, 1}};
        auto x = mm->add_parameter("x", sd);
        auto w = mm->add_parameter("w", sw);

        auto* then_mod = p.create_module("If_6_if");
        auto out1      = then_mod->add_instruction(migraphx::make_op("dot"), a, b);
        then_mod->add_return({out1});

        auto* else_mod = p.create_module("If_6_else");
        auto out2      = else_mod->add_instruction(migraphx::make_op("convolution"), x, w);
        else_mod->add_return({out2});

        auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        mm->add_return({ret});

        return p;
    };

    auto create_program_op = [&] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sx{migraphx::shape::float_type, {2, 2, 4, 8}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 2, 8, 6}};
        migraphx::shape sc{migraphx::shape::bool_type};
        auto cond = mm->add_parameter("cond", sc);
        auto a    = mm->add_parameter("a", sx);
        auto b    = mm->add_parameter("b", sy);

        migraphx::shape sd{migraphx::shape::float_type, {2, 2, 4, 6}};
        migraphx::shape sw{migraphx::shape::float_type, {2, 2, 1, 1}};
        auto x = mm->add_parameter("x", sd);
        auto w = mm->add_parameter("w", sw);

        auto* then_mod = p.create_module("If_6_if");
        auto ca   = then_mod->add_instruction(migraphx::make_op("capture", {{"ins_index", 2}}), a);
        auto cb   = then_mod->add_instruction(migraphx::make_op("capture", {{"ins_index", 3}}), b);
        auto out1 = then_mod->add_instruction(migraphx::make_op("dot"), ca, cb);
        then_mod->add_return({out1});

        auto* else_mod = p.create_module("If_6_else");
        auto cx   = else_mod->add_instruction(migraphx::make_op("capture", {{"ins_index", 0}}), x);
        auto cw   = else_mod->add_instruction(migraphx::make_op("capture", {{"ins_index", 1}}), w);
        auto out2 = else_mod->add_instruction(migraphx::make_op("convolution"), cx, cw);
        else_mod->add_return({out2});

        auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        mm->add_return({ret});

        return p;
    };

    {
        auto p                  = create_program();
        auto op_capture_p       = create_program_op();
        migraphx::target t      = migraphx::make_target("ref");
        std::size_t param_index = 0;
        migraphx::run_passes(
            p, {migraphx::capture_arguments_pass{{"dot", "convolution"}, {}, &param_index}});

        EXPECT(p == op_capture_p);
    }
}

TEST_CASE(dot_float)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);

        auto r = migraphx::add_apply_alpha_beta(*mm, {pa, pb}, migraphx::make_op("dot"));
        mm->add_return({r});

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa      = mm->add_parameter("a", sa);
        auto pb      = mm->add_parameter("b", sb);
        auto zp_a    = mm->add_literal(static_cast<int8_t>(0));
        auto scale_a = mm->add_literal(10.0f);
        scale_a      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}), scale_a);
        zp_a = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}),
                                   zp_a);
        auto qa  = mm->add_instruction(migraphx::make_op("quantizelinear"), pa, scale_a, zp_a);
        auto dqa = mm->add_instruction(migraphx::make_op("dequantizelinear"), qa, scale_a, zp_a);

        auto zp_b    = mm->add_literal(static_cast<int8_t>(0));
        auto scale_b = mm->add_literal(10.0f);
        scale_b      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sb.lens()}}), scale_b);
        zp_b = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sb.lens()}}),
                                   zp_b);
        auto qb  = mm->add_instruction(migraphx::make_op("quantizelinear"), pb, scale_b, zp_b);
        auto dqb = mm->add_instruction(migraphx::make_op("dequantizelinear"), qb, scale_b, zp_b);

        auto r = migraphx::add_apply_alpha_beta(*mm, {dqa, dqb}, migraphx::make_op("dot"));
        mm->add_return({r});

        return p;
    };

    auto create_int8_optimized_prog = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa      = mm->add_parameter("a", sa);
        auto pb      = mm->add_parameter("b", sb);
        auto zp      = mm->add_literal(static_cast<int8_t>(0));
        auto scale   = mm->add_literal(10.0f);
        auto scale_a = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}), scale);
        auto zp_a =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}), zp);
        auto quant_a = mm->add_instruction(migraphx::make_op("quantizelinear"), pa, scale_a, zp_a);
        auto scale_b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sb.lens()}}), scale);
        auto zp_b =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sb.lens()}}), zp);
        auto quant_b  = mm->add_instruction(migraphx::make_op("quantizelinear"), pb, scale_b, zp_b);
        auto quant    = mm->add_instruction(migraphx::make_op("quant_dot"), quant_a, quant_b);
        auto scale_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", quant->get_shape().lens()}}), scale);
        auto out_scale = mm->add_instruction(migraphx::make_op("mul"), scale_mb, scale_mb);
        auto r = mm->add_instruction(migraphx::make_op("dequantizelinear"), quant, out_scale);
        mm->add_return({r});

        return p;
    };

    const std::vector<std::pair<float, float>> quant_params = {
        {0.1f, 0.0f}, {0.1f, 0.0f}, {0.1f, 100.0f}};
    auto p                  = create_program();
    std::size_t param_index = 0;
    migraphx::run_passes(p, {migraphx::capture_arguments_pass{{"dot"}, {}, &param_index}});
    migraphx::run_passes(
        p,
        {migraphx::quantize_8bits_pass{migraphx::shape::type_t::int8_type, quant_params},
         migraphx::dead_code_elimination{}});
    auto qp = create_int8_quantized_prog();

    EXPECT(p.sort() == qp.sort());

    optimize_prog_int8(p);
    auto op = create_int8_optimized_prog();
    EXPECT(p.sort() == op.sort());
}

TEST_CASE(dot_double_2args)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::double_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::double_type, {16, 8}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);
        auto r  = migraphx::add_apply_alpha_beta(*mm, {pa, pb}, migraphx::make_op("dot"));
        mm->add_return({r});

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::double_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::double_type, {16, 8}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);

        auto zp_a    = mm->add_literal(static_cast<int8_t>(0));
        auto scale_a = mm->add_literal(10.0);
        scale_a      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}), scale_a);
        zp_a = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}),
                                   zp_a);
        auto qa   = mm->add_instruction(migraphx::make_op("quantizelinear"), pa, scale_a, zp_a);
        auto dqa  = mm->add_instruction(migraphx::make_op("dequantizelinear"), qa, scale_a, zp_a);
        auto zp_b = mm->add_literal(static_cast<int8_t>(0));
        auto scale_b = mm->add_literal(5.0);
        scale_b      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sb.lens()}}), scale_b);
        zp_b = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sb.lens()}}),
                                   zp_b);
        auto qb  = mm->add_instruction(migraphx::make_op("quantizelinear"), pb, scale_b, zp_b);
        auto dqb = mm->add_instruction(migraphx::make_op("dequantizelinear"), qb, scale_b, zp_b);
        auto r   = migraphx::add_apply_alpha_beta(*mm, {dqa, dqb}, migraphx::make_op("dot"));
        mm->add_return({r});
        return p;
    };

    auto create_int8_optimized_prog = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::double_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::double_type, {16, 8}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);

        auto scale_a_lit = mm->add_literal(10.0);
        auto zp          = mm->add_literal(static_cast<int8_t>(0));
        auto scale_a     = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}), scale_a_lit);
        auto zp_a =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}), zp);
        auto qa = mm->add_instruction(migraphx::make_op("quantizelinear"), pa, scale_a, zp_a);
        auto scale_b_lit = mm->add_literal(5.0);
        auto scale_b     = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sb.lens()}}), scale_b_lit);
        auto zp_b =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sb.lens()}}), zp);
        auto qb   = mm->add_instruction(migraphx::make_op("quantizelinear"), pb, scale_b, zp_b);
        auto qdot = mm->add_instruction(migraphx::make_op("quant_dot"), qa, qb);
        auto scale_a_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", qdot->get_shape().lens()}}),
            scale_a_lit);
        auto scale_b_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", qdot->get_shape().lens()}}),
            scale_b_lit);
        auto out_scale = mm->add_instruction(migraphx::make_op("mul"), scale_a_mb, scale_b_mb);
        auto r = mm->add_instruction(migraphx::make_op("dequantizelinear"), qdot, out_scale);
        mm->add_return({r});
        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{{0.1f, 0.0f}, {0.2f, 0.0f}};
    std::size_t param_index = 0;
    migraphx::run_passes(p, {migraphx::capture_arguments_pass{{"dot"}, {}, &param_index}});
    migraphx::run_passes(
        p,
        {migraphx::quantize_8bits_pass{migraphx::shape::type_t::int8_type, quant_params},
         migraphx::dead_code_elimination{}});
    EXPECT(p == create_int8_quantized_prog());

    optimize_prog_int8(p);
    EXPECT(p == create_int8_optimized_prog());
}

TEST_CASE(dot_half_1arg)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::half_type, {9, 9}};
        auto x = mm->add_parameter("x", s);
        auto r = mm->add_instruction(migraphx::make_op("dot"), x, x);
        mm->add_return({r});

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::half_type, {9, 9}};
        auto x = mm->add_parameter("x", sa);

        auto zp_a    = mm->add_literal(static_cast<int8_t>(0));
        auto scale_a = mm->add_literal(migraphx::literal({sa.type()}, {10.0}));
        scale_a      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}), scale_a);
        zp_a = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}),
                                   zp_a);
        auto qa   = mm->add_instruction(migraphx::make_op("quantizelinear"), x, scale_a, zp_a);
        auto dqa  = mm->add_instruction(migraphx::make_op("dequantizelinear"), qa, scale_a, zp_a);
        auto zp_b = mm->add_literal(static_cast<int8_t>(0));
        auto scale_b = mm->add_literal(migraphx::literal({sa.type()}, {10.0}));
        scale_b      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}), scale_b);
        zp_b = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}),
                                   zp_b);
        auto qb  = mm->add_instruction(migraphx::make_op("quantizelinear"), x, scale_b, zp_b);
        auto dqb = mm->add_instruction(migraphx::make_op("dequantizelinear"), qb, scale_b, zp_b);
        auto r   = mm->add_instruction(migraphx::make_op("dot"), dqa, dqb);
        mm->add_return({r});
        return p;
    };

    auto create_int8_optimized_prog = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::half_type, {9, 9}};
        auto x = mm->add_parameter("x", sa);

        auto zp        = mm->add_literal(static_cast<int8_t>(0));
        auto scale_lit = mm->add_literal(migraphx::literal({sa.type()}, {10.0}));
        auto scale     = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}), scale_lit);
        zp =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}), zp);
        auto qx        = mm->add_instruction(migraphx::make_op("quantizelinear"), x, scale, zp);
        auto qdot      = mm->add_instruction(migraphx::make_op("quant_dot"), qx, qx);
        auto out_scale = mm->add_instruction(migraphx::make_op("mul"), scale, scale);
        auto r = mm->add_instruction(migraphx::make_op("dequantizelinear"), qdot, out_scale);
        mm->add_return({r});
        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{{0.1f, 0.0f}, {0.1f, 0.0f}};
    std::size_t param_index = 0;
    migraphx::run_passes(p, {migraphx::capture_arguments_pass{{"dot"}, {}, &param_index}});
    migraphx::run_passes(p,
                         {migraphx::quantize_8bits_pass{migraphx::shape::int8_type, quant_params},
                          migraphx::dead_code_elimination{}});
    EXPECT(p == create_int8_quantized_prog());

    optimize_prog_int8(p);
    EXPECT(p == create_int8_optimized_prog());
}

TEST_CASE(conv_float)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto input =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto weights =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto r = mm->add_instruction(migraphx::make_op("convolution"), input, weights);
        mm->add_return({r});

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sx{migraphx::shape::float_type, {4, 3, 3, 3}};
        migraphx::shape sw{migraphx::shape::float_type, {4, 3, 3, 3}};
        auto px = mm->add_parameter("x", sx);
        auto pw = mm->add_parameter("w", sw);

        auto zp        = mm->add_literal(static_cast<int8_t>(0));
        auto scale_lit = mm->add_literal(10.0f);
        auto scale     = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sx.lens()}}), scale_lit);
        zp =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sx.lens()}}), zp);
        auto quant_x = mm->add_instruction(migraphx::make_op("quantizelinear"), px, scale, zp);
        auto quant_w = mm->add_instruction(migraphx::make_op("quantizelinear"), pw, scale, zp);

        auto quant = mm->add_instruction(migraphx::make_op("quant_convolution"), quant_x, quant_w);

        auto scale_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", quant->get_shape().lens()}}),
            scale_lit);
        auto out_scale = mm->add_instruction(migraphx::make_op("mul"), scale_mb, scale_mb);
        auto r = mm->add_instruction(migraphx::make_op("dequantizelinear"), quant, out_scale);
        mm->add_return({r});

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{{0.1f, 0.0f}, {0.1f, 0.0f}};
    std::size_t param_index = 0;
    migraphx::run_passes(p, {migraphx::capture_arguments_pass{{"convolution"}, {}, &param_index}});
    migraphx::run_passes(
        p, {migraphx::quantize_8bits_pass{migraphx::shape::type_t::int8_type, quant_params}});
    optimize_prog_int8(p);
    auto qp = create_int8_quantized_prog();

    EXPECT(p == qp);
}

TEST_CASE(conv_float_throw)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto input =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto weights =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto r = mm->add_instruction(migraphx::make_op("convolution"), input, weights);
        mm->add_return({r});

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{{0.1f, 0.0f}, {0.1f, 0.0f}};
    test::throws([&] {
        migraphx::run_passes(
            p, {migraphx::quantize_8bits_pass{migraphx::shape::type_t::int8_type, quant_params}});
    });
}

TEST_CASE(conv_half)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto input =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::half_type, {4, 3, 3, 3}});
        auto weights =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::half_type, {4, 3, 3, 3}});
        auto r = mm->add_instruction(migraphx::make_op("convolution"), input, weights);
        mm->add_return({r});

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sx{migraphx::shape::half_type, {4, 3, 3, 3}};
        migraphx::shape sw{migraphx::shape::half_type, {4, 3, 3, 3}};
        auto px = mm->add_parameter("x", sx);
        auto pw = mm->add_parameter("w", sw);

        auto zp        = mm->add_literal(static_cast<int8_t>(0));
        auto scale_lit = mm->add_literal(migraphx::literal({sx.type()}, {10.0}));
        auto scale     = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sx.lens()}}), scale_lit);
        zp =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sx.lens()}}), zp);
        auto quant_x = mm->add_instruction(migraphx::make_op("quantizelinear"), px, scale, zp);
        auto quant_w = mm->add_instruction(migraphx::make_op("quantizelinear"), pw, scale, zp);

        auto quant = mm->add_instruction(migraphx::make_op("quant_convolution"), quant_x, quant_w);
        auto scale_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", quant->get_shape().lens()}}),
            scale_lit);
        auto out_scale = mm->add_instruction(migraphx::make_op("mul"), scale_mb, scale_mb);
        auto r = mm->add_instruction(migraphx::make_op("dequantizelinear"), quant, out_scale);
        mm->add_return({r});

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{{0.1f, 0.0f}, {0.1f, 0.0f}};
    std::size_t param_index = 0;
    migraphx::run_passes(p, {migraphx::capture_arguments_pass{{"convolution"}, {}, &param_index}});
    migraphx::run_passes(
        p, {migraphx::quantize_8bits_pass{migraphx::shape::type_t::int8_type, quant_params}});
    optimize_prog_int8(p);
    auto qp = create_int8_quantized_prog();

    EXPECT(p == qp);
}

template <class T>
static auto get_hash(const T& x)
{
    return std::hash<T>{}(x);
}

TEST_CASE(target_copy)
{
    auto run_prog = [](migraphx::program p,
                       const migraphx::target& t,
                       migraphx::parameter_map& m_in,
                       std::vector<float>& res) {
        p.compile(t);
        migraphx::parameter_map m;
        for(auto&& x : p.get_parameter_shapes())
        {
            if(m_in.count(x.first) > 0)
            {
                m[x.first] = t.copy_to(m_in[x.first]);
            }
            else
            {
                m[x.first] = t.allocate(x.second);
            }
        }

        auto result = t.copy_from(p.eval(m).back());
        result.visit([&](auto v) { res.assign(v.begin(), v.end()); });
    };

    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto p1 = mm->add_parameter("x", s);
        auto p2 = mm->add_parameter("y", s);
        mm->add_instruction(migraphx::make_op("add"), p1, p2);

        return p;
    };

    {
        auto p = create_program();
        migraphx::parameter_map m;
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        m["x"] = migraphx::generate_argument(s);
        std::vector<float> ref_result;
        migraphx::target ref_t = migraphx::make_target("ref");
        run_prog(p, ref_t, m, ref_result);

        std::vector<float> orig_result;
        run_prog(p, ref_t, m, orig_result);

        EXPECT(migraphx::verify::verify_rms_range(ref_result, orig_result));
    }
}

TEST_CASE(int8_quantization_dot)
{
    auto run_prog = [](migraphx::program p,
                       const migraphx::target& t,
                       migraphx::parameter_map& m_in,
                       std::vector<float>& res,
                       bool b_quantize = false) {
        if(b_quantize)
        {
            std::vector<migraphx::parameter_map> cali_data;
            cali_data.push_back(m_in);
            migraphx::quantize_int8(p, t, cali_data);
        }
        p.compile(t);
        migraphx::parameter_map m;
        for(auto&& x : p.get_parameter_shapes())
        {
            if(m_in.count(x.first) > 0)
            {
                m[x.first] = t.copy_to(m_in[x.first]);
            }
            else
            {
                m[x.first] = t.allocate(x.second);
            }
        }

        auto result = t.copy_from(p.eval(m).back());
        result.visit([&](auto v) { res.assign(v.begin(), v.end()); });
    };

    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);
        auto pc = mm->add_parameter("c", sc);
        auto r =
            migraphx::add_apply_alpha_beta(*mm, {pa, pb, pc}, migraphx::make_op("dot"), 1.0f, 1.0f);
        mm->add_return({r});
        return p;
    };

    {
        auto p = create_program();
        migraphx::parameter_map m;
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        m["a"] = migraphx::generate_argument(sa, get_hash(std::string("a")));
        m["b"] = migraphx::generate_argument(sb, get_hash(std::string("b")));
        std::vector<float> quant_result;
        migraphx::target ref_t = migraphx::make_target("ref");
        run_prog(p, ref_t, m, quant_result, true);

        std::vector<float> no_quant_result;
        run_prog(p, ref_t, m, no_quant_result);

        EXPECT(migraphx::verify::verify_range_with_tolerance(
            quant_result,
            migraphx::verify::expected{no_quant_result},
            migraphx::verify::tolerance{0.003}));
    }
}

TEST_CASE(int8_quantization_conv)
{
    auto run_prog = [](migraphx::program p,
                       const migraphx::target& t,
                       std::vector<float>& res,
                       bool b_quantize = false) {
        if(b_quantize)
        {
            std::vector<migraphx::parameter_map> cali_data;
            migraphx::quantize_int8(p, t, cali_data);
        }
        p.compile(t);
        migraphx::parameter_map m;

        auto result = t.copy_from(p.eval(m).back());
        result.visit([&](auto v) { res.assign(v.begin(), v.end()); });
    };

    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sx{migraphx::shape::float_type, {4, 2, 2, 2}};
        migraphx::shape sw{migraphx::shape::float_type, {4, 2, 2, 2}};
        std::vector<float> v(sx.elements(), 0.5f);
        auto input   = mm->add_literal(migraphx::literal(sx, v));
        auto weights = mm->add_literal(migraphx::literal(sw, v));
        auto r       = mm->add_instruction(migraphx::make_op("convolution"), input, weights);
        mm->add_return({r});

        return p;
    };

    {
        auto p = create_program();
        std::vector<float> quant_result;
        migraphx::target ref_t = migraphx::make_target("ref");
        run_prog(p, ref_t, quant_result, true);

        std::vector<float> no_quant_result;
        run_prog(p, ref_t, no_quant_result);

        EXPECT(migraphx::verify::verify_rms_range(quant_result, no_quant_result));
    }
}

TEST_CASE(int8_subgraph)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sx{migraphx::shape::float_type, {2, 2, 4, 8}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 2, 8, 6}};
        migraphx::shape sc{migraphx::shape::bool_type};
        auto cond = mm->add_parameter("cond", sc);
        auto a    = mm->add_parameter("a", sx);
        auto b    = mm->add_parameter("b", sy);

        migraphx::shape sd{migraphx::shape::float_type, {2, 2, 4, 6}};
        migraphx::shape sw{migraphx::shape::float_type, {2, 2, 1, 1}};
        auto x = mm->add_parameter("x", sd);
        auto w = mm->add_parameter("w", sw);

        auto* then_mod = p.create_module("If_6_if");
        auto out1 = migraphx::add_apply_alpha_beta(*then_mod, {a, b}, migraphx::make_op("dot"));
        then_mod->add_return({out1});

        auto* else_mod = p.create_module("If_6_else");
        auto out2      = else_mod->add_instruction(migraphx::make_op("convolution"), x, w);
        else_mod->add_return({out2});

        auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        mm->add_return({ret});

        return p;
    };

    auto create_int8_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape sx{migraphx::shape::float_type, {2, 2, 4, 8}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 2, 8, 6}};
        migraphx::shape sout{migraphx::shape::float_type, {2, 2, 4, 6}};
        migraphx::shape sc{migraphx::shape::bool_type};
        auto cond = mm->add_parameter("cond", sc);
        auto a    = mm->add_parameter("a", sx);
        auto b    = mm->add_parameter("b", sy);

        // then submod
        auto* then_mod = p.create_module("If_6_if");
        auto zp1       = then_mod->add_literal(static_cast<int8_t>(0));
        auto s1        = then_mod->add_literal(10.0f);
        auto sa        = then_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sx.lens()}}), s1);
        auto zpa = then_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sx.lens()}}), zp1);
        auto qa = then_mod->add_instruction(migraphx::make_op("quantizelinear"), a, sa, zpa);
        auto sb = then_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sy.lens()}}), s1);
        auto zpb = then_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sy.lens()}}), zp1);
        auto qb    = then_mod->add_instruction(migraphx::make_op("quantizelinear"), b, sb, zpb);
        auto qdot  = then_mod->add_instruction(migraphx::make_op("quant_dot"), qa, qb);
        auto s1_mb = then_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", qdot->get_shape().lens()}}), s1);
        auto so     = then_mod->add_instruction(migraphx::make_op("mul"), s1_mb, s1_mb);
        auto r      = then_mod->add_instruction(migraphx::make_op("dequantizelinear"), qdot, so);
        then_mod->add_return({r});

        migraphx::shape sd{migraphx::shape::float_type, {2, 2, 4, 6}};
        migraphx::shape sw{migraphx::shape::float_type, {2, 2, 1, 1}};
        auto x = mm->add_parameter("x", sd);
        auto w = mm->add_parameter("w", sw);
        // else submod
        auto* else_mod = p.create_module("If_6_else");
        auto sax_lit   = else_mod->add_literal(2.0f);
        auto zp        = else_mod->add_literal(static_cast<int8_t>(0));
        auto sax       = else_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sd.lens()}}), sax_lit);
        auto zpx = else_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sd.lens()}}), zp);
        auto qx      = else_mod->add_instruction(migraphx::make_op("quantizelinear"), x, sax, zpx);
        auto ssw_lit = else_mod->add_literal(1.66667f);
        auto ssw     = else_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sw.lens()}}), ssw_lit);
        auto zpw = else_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sw.lens()}}), zp);
        auto qw     = else_mod->add_instruction(migraphx::make_op("quantizelinear"), w, ssw, zpw);
        auto qconv  = else_mod->add_instruction(migraphx::make_op("quant_convolution"), qx, qw);
        auto ssw_mb = else_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", qconv->get_shape().lens()}}),
            ssw_lit);
        auto so1 = else_mod->add_instruction(migraphx::make_op("mul"), sax, ssw_mb);
        auto r1  = else_mod->add_instruction(migraphx::make_op("dequantizelinear"), qconv, so1);
        else_mod->add_return({r1});

        auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        mm->add_return({ret});

        return p;
    };

    auto p1 = create_program();
    const std::vector<std::pair<float, float>>& quant_params{
        {0.5f, 0.0f}, {0.6f, 0.0f}, {0.1f, 0.0f}, {0.1f, 0.0f}};
    std::size_t param_index = 0;
    migraphx::run_passes(
        p1, {migraphx::capture_arguments_pass{{"convolution", "dot"}, {}, &param_index}});
    migraphx::run_passes(
        p1, {migraphx::quantize_8bits_pass{migraphx::shape::type_t::int8_type, quant_params}});
    optimize_prog_int8(p1);

    auto p2 = create_int8_program();
    EXPECT(p1 == p2);
}

TEST_CASE(test_op_capture)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s1{migraphx::shape::float_type, {3, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 6}};
    std::vector<float> d1(s1.elements());
    std::vector<float> d2(s2.elements());
    std::iota(d1.begin(), d1.end(), 0.0f);
    std::iota(d2.begin(), d2.end(), 0.0f);

    auto p1 = mm->add_literal(s1, d1);
    auto p2 = mm->add_literal(s1, d1);
    auto pb = mm->add_literal(s2, d2);
    auto pc = mm->add_literal(s2, d2);
    auto pa = mm->add_instruction(migraphx::make_op("add"), p1, p2);
    auto ps =
        migraphx::add_apply_alpha_beta(*mm, {pa, pb, pc}, migraphx::make_op("dot"), 1.0f, 1.0f);
    mm->add_instruction(migraphx::make_op("dot"), pa, ps);

    auto calc = [](std::size_t, const std::vector<migraphx::argument>&) {};

    migraphx::program capture_p = p;
    migraphx::target t          = migraphx::make_target("ref");
    std::size_t param_index     = 0;
    migraphx::run_passes(capture_p,
                         {migraphx::capture_arguments_pass{{"dot"}, calc, &param_index}});

    p.compile(migraphx::make_target("ref"));
    capture_p.compile(migraphx::make_target("ref"));

    auto cap_res = capture_p.eval({}).back();
    auto res     = p.eval({}).back();

    std::vector<float> vec;
    std::vector<float> cap_vec;
    cap_res.visit([&](auto output) { cap_vec.assign(output.begin(), output.end()); });
    res.visit([&](auto output) { vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(vec, cap_vec));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
