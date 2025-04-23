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
#include <numeric>
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

TEST_CASE(add_literals)
{
    migraphx::program p;
    migraphx::module m = p.get_main_module();
    migraphx::shape param_shape{migraphx_shape_float_type, {3, 3}};
    std::vector<float> x_values(9, 1);
    auto x = m.add_literal(param_shape, x_values.data());
    std::vector<float> y_values(9, -1);
    auto y      = m.add_literal(param_shape, y_values.data());
    auto add_op = migraphx::operation("add");
    auto r      = m.add_instruction(add_op, {x, y});
    m.add_return({r});
    // run on ref target
    p.compile(migraphx::target("ref"));
    migraphx::program_parameters pp;
    auto outputs = p.eval(pp);
    auto output  = outputs[0];
    std::vector<float> expected(9, 0);
    CHECK(bool(output == migraphx::argument(param_shape, expected.data())));
}

TEST_CASE(if_then_else_op)
{
    migraphx::shape param_shape{migraphx_shape_float_type, {3, 3}};
    migraphx::shape cond_s{migraphx_shape_bool_type};
    auto create_program = [&]() {
        migraphx::program p;
        auto mm         = p.get_main_module();
        auto cond       = mm.add_parameter("cond", cond_s);
        auto x          = mm.add_parameter("x", param_shape);
        auto y          = mm.add_parameter("y", param_shape);
        auto then_mod   = p.create_module("If_0_if");
        auto x_identity = then_mod.add_instruction(migraphx::operation("identity"), {x});
        then_mod.add_return({x_identity});

        auto else_mod   = p.create_module("If_0_else");
        auto y_identity = else_mod.add_instruction(migraphx::operation("identity"), {y});
        else_mod.add_return({y_identity});

        auto if_ins = mm.add_instruction(migraphx::operation("if"), {cond}, {then_mod, else_mod});
        auto get_tuple_op = migraphx::operation("get_tuple_elem", "{index: 0}");
        auto ret          = mm.add_instruction(get_tuple_op, {if_ins});
        mm.add_return({ret});
        return p;
    };

    std::vector<float> x_data(9, 1);
    std::vector<float> y_data(9, -1);
    auto x_arg    = migraphx::argument(param_shape, x_data.data());
    auto y_arg    = migraphx::argument(param_shape, y_data.data());
    auto run_prog = [&](bool cond) {
        auto p = create_program();
        p.compile(migraphx::target("ref"));
        auto outputs =
            p.eval({{"cond", migraphx::argument(cond_s, &cond)}, {"x", x_arg}, {"y", y_arg}});
        return outputs[0];
    };

    // then branch
    auto then_res = run_prog(true);
    CHECK(bool{then_res == x_arg});

    // else branch
    auto else_res = run_prog(false);
    CHECK(bool{else_res == y_arg});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
