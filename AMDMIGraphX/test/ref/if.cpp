/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(if_literal_test)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape cond_s{migraphx::shape::bool_type};
        auto cond = mm->add_parameter("cond", cond_s);

        migraphx::shape s{migraphx::shape::float_type, {5}};

        auto* then_mod           = p.create_module("If_0_if");
        std::vector<float> data1 = {1, 2, 3, 4, 5};
        auto l1                  = then_mod->add_literal(migraphx::literal(s, data1));
        then_mod->add_return({l1});

        auto* else_mod           = p.create_module("If_0_else");
        std::vector<float> data2 = {5, 4, 3, 2, 1};
        auto l2                  = else_mod->add_literal(migraphx::literal(s, data2));
        else_mod->add_return({l2});

        auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        auto r   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
        mm->add_return({r});

        return p;
    };

    auto run_prog = [&](bool cond) {
        auto p = create_program();
        p.compile(migraphx::make_target("ref"));
        std::vector<char> c_data = {static_cast<char>(cond)};
        migraphx::shape cs{migraphx::shape::bool_type};
        migraphx::parameter_map m;
        m["cond"] = migraphx::argument(cs, c_data.data());

        auto res = p.eval(m).back();
        std::vector<float> ret;
        res.visit([&](auto v) { ret.assign(v.begin(), v.end()); });

        return ret;
    };

    // then branch
    {
        std::vector<float> gold_ret = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        auto ret                    = run_prog(true);
        EXPECT(gold_ret == ret);
    }

    // else branch
    {
        std::vector<float> gold_ret = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        auto ret                    = run_prog(false);
        EXPECT(gold_ret == ret);
    }
}

TEST_CASE(if_param_test)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape cond_s{migraphx::shape::bool_type};
        auto cond = mm->add_parameter("cond", cond_s);
        migraphx::shape ds{migraphx::shape::float_type, {2, 3}};
        auto x                   = mm->add_parameter("x", ds);
        auto y                   = mm->add_parameter("y", ds);
        std::vector<float> data2 = {-0.258047, 0.360394, 0.536804, -0.577762, 1.0217, 1.02442};
        auto l2                  = mm->add_literal(migraphx::literal(ds, data2));
        auto sum                 = mm->add_instruction(migraphx::make_op("add"), x, l2);

        auto* then_mod           = p.create_module("If_0_if");
        std::vector<float> data1 = {0.384804, -1.77948, -0.453775, 0.477438, -1.06333, -1.12893};
        auto l1                  = then_mod->add_literal(migraphx::literal(ds, data1));
        auto tx                  = then_mod->add_parameter("x", ds);
        auto a1                  = then_mod->add_instruction(migraphx::make_op("add"), tx, l1);
        then_mod->add_return({a1});

        auto* else_mod = p.create_module("If_0_else");
        auto ey        = else_mod->add_parameter("y", ds);
        auto a2        = else_mod->add_instruction(migraphx::make_op("mul"), ey, sum);
        else_mod->add_return({a2});

        auto ret = mm->add_instruction(migraphx::make_op("if"), {cond, x, y}, {then_mod, else_mod});
        auto r   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
        mm->add_return({r});

        return p;
    };

    auto run_prog = [&](bool cond) {
        auto p = create_program();
        p.compile(migraphx::make_target("ref"));
        std::vector<char> c_data = {static_cast<char>(cond)};
        migraphx::shape cs{migraphx::shape::bool_type};
        migraphx::parameter_map m;
        m["cond"] = migraphx::argument(cs, c_data.data());
        migraphx::shape ds{migraphx::shape::float_type, {2, 3}};
        std::vector<float> data_x(ds.elements(), 1);
        m["x"] = migraphx::argument(ds, data_x.data());
        std::vector<float> data_y(ds.elements(), 2);
        m["y"] = migraphx::argument(ds, data_y.data());

        auto res = p.eval(m).back();
        std::vector<float> ret;
        res.visit([&](auto v) { ret.assign(v.begin(), v.end()); });
        return ret;
    };

    // then branch
    {
        std::vector<float> gold_ret = {
            1.384804, -0.77947998, 0.54622501, 1.477438, -0.063330054, -0.12892997};
        auto ret = run_prog(true);
        EXPECT(gold_ret == ret);
    }

    // else branch
    {
        std::vector<float> gold_ret = {
            1.483906, 2.720788, 3.0736079, 0.84447598, 4.0433998, 4.04884};
        auto ret = run_prog(false);
        EXPECT(gold_ret == ret);
    }
}

TEST_CASE(if_pl_test)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape cond_s{migraphx::shape::bool_type};
        migraphx::shape s{migraphx::shape::float_type, {5}};
        auto cond = mm->add_parameter("cond", cond_s);
        auto x    = mm->add_parameter("x", s);

        auto* then_mod           = p.create_module("If_0_if");
        std::vector<float> data1 = {1, 2, 3, 4, 5};
        auto l1                  = then_mod->add_literal(migraphx::literal(s, data1));
        then_mod->add_return({l1, x});

        auto* else_mod           = p.create_module("If_0_else");
        std::vector<float> data2 = {5, 4, 3, 2, 1};
        auto l2                  = else_mod->add_literal(migraphx::literal(s, data2));
        auto s2                  = else_mod->add_instruction(migraphx::make_op("add"), x, l2);
        else_mod->add_return({s2, l2});

        auto ret     = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        auto outline = mm->add_outline(s);
        auto r = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
        mm->add_return({outline, r});

        return p;
    };

    auto run_prog = [&](bool cond) {
        auto p = create_program();
        p.compile(migraphx::make_target("ref"));
        std::vector<char> c_data = {static_cast<char>(cond)};
        migraphx::shape cs{migraphx::shape::bool_type};
        migraphx::parameter_map m;
        m["cond"] = migraphx::argument(cs, c_data.data());
        migraphx::shape ds{migraphx::shape::float_type, {5}};
        std::vector<float> data(ds.elements(), 1);
        m["x"] = migraphx::argument(ds, data.data());

        auto res = p.eval(m).back();
        std::vector<float> ret;
        res.visit([&](auto v) { ret.assign(v.begin(), v.end()); });

        return ret;
    };

    // then branch
    {
        std::vector<float> gold_ret = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        auto ret                    = run_prog(true);
        EXPECT(gold_ret == ret);
    }

    // else branch
    {
        std::vector<float> gold_ret = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f};
        auto ret                    = run_prog(false);
        EXPECT(gold_ret == ret);
    }
}
