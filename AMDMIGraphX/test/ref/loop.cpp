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
#include <migraphx/quantization.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include "test.hpp"

static auto run_prog(int64_t iter_num, bool cond, int64_t ini_val)
{
    migraphx::shape si{migraphx::shape::int64_type};
    migraphx::shape s{migraphx::shape::int64_type, {1}};
    migraphx::shape sc{migraphx::shape::bool_type};

    auto create_program = [&]() {
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto in_iter = mm->add_parameter("iter_num", si);
        auto in_cond = mm->add_parameter("ccond", sc);
        auto in_val  = mm->add_parameter("val", s);

        auto* body = p.create_module("loop_module");
        auto iter  = body->add_parameter("#loop_module_in_0", si);
        body->add_parameter("#loop_module_in_1", sc);
        auto in_v               = body->add_parameter("#loop_module_in_2", s);
        std::vector<int64_t> vd = {3};
        auto l                  = body->add_literal(migraphx::literal(si, vd));
        auto ad                 = body->add_instruction(migraphx::make_op("add"), iter, l);
        auto val                = body->add_instruction(migraphx::make_op("add"), in_v, ad);
        auto eq                 = body->add_instruction(migraphx::make_op("equal"), iter, l);
        auto beq                = body->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), eq);
        auto neq = body->add_instruction(migraphx::make_op("not"), beq);
        body->add_return({neq, val, val});

        auto rl = mm->add_instruction(migraphx::make_op("loop", {{"max_iterations", 10}}),
                                      {in_iter, in_cond, in_val},
                                      {body});
        auto r0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), rl);
        auto r1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), rl);
        mm->add_return({r0, r1});

        return p;
    };

    auto p = create_program();
    p.compile(migraphx::make_target("ref"));
    migraphx::parameter_map pp;
    pp["iter_num"] = migraphx::argument(si, &iter_num);
    pp["ccond"]    = migraphx::argument(sc, &cond);
    pp["val"]      = migraphx::argument(s, &ini_val);
    auto rets      = p.eval(pp);

    std::vector<std::vector<int64_t>> res;
    for(auto& arg : rets)
    {
        std::vector<int64_t> vec;
        arg.visit([&](auto v) { vec.assign(v.begin(), v.end()); });
        res.push_back(vec);
    }

    return res;
}

TEST_CASE(loop_test1)
{
    auto ress                      = run_prog(10, true, 1);
    std::vector<int64_t> gold_last = {19};
    EXPECT(ress.front() == gold_last);
    std::vector<int64_t> gold_concat = {4, 8, 13, 19, 0, 0, 0, 0, 0, 0};
    EXPECT(ress.back() == gold_concat);
}

TEST_CASE(loop_test2)
{
    auto ress                      = run_prog(4, true, 1);
    std::vector<int64_t> gold_last = {19};
    EXPECT(ress.front() == gold_last);
    std::vector<int64_t> gold_concat = {4, 8, 13, 19, 0, 0, 0, 0, 0, 0};
    EXPECT(ress.back() == gold_concat);
}

TEST_CASE(loop_test3)
{
    auto ress                      = run_prog(3, true, 1);
    std::vector<int64_t> gold_last = {13};
    EXPECT(ress.front() == gold_last);
    std::vector<int64_t> gold_concat = {4, 8, 13, 0, 0, 0, 0, 0, 0, 0};
    EXPECT(ress.back() == gold_concat);
}

TEST_CASE(loop_test4)
{
    auto ress                      = run_prog(5, true, 2);
    std::vector<int64_t> gold_last = {20};
    EXPECT(ress.front() == gold_last);
    std::vector<int64_t> gold_concat = {5, 9, 14, 20, 0, 0, 0, 0, 0, 0};
    EXPECT(ress.back() == gold_concat);
}
