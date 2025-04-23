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

#include "verify_program.hpp"
#include <migraphx/literal.hpp>
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_loop : verify_program<test_loop>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape si{migraphx::shape::int64_type};
        migraphx::shape s{migraphx::shape::int64_type, {1}};
        migraphx::shape sc{migraphx::shape::bool_type};
        int64_t iter_num = 10;
        auto in_iter     = mm->add_literal(migraphx::literal(si, {iter_num}));
        auto in_cond     = mm->add_parameter("ccond", sc);
        int64_t value    = 5;
        auto in_val      = mm->add_literal(migraphx::literal(s, {value}));

        auto* body = p.create_module("loop_module");
        auto iter  = body->add_parameter("iter_num", si);
        body->add_parameter("cond", sc);
        auto in_v               = body->add_parameter("input", s);
        std::vector<int64_t> vd = {3};
        auto l                  = body->add_literal(migraphx::literal(si, vd));
        auto ad                 = body->add_instruction(migraphx::make_op("add"), iter, l);
        auto val                = body->add_instruction(migraphx::make_op("add"), in_v, ad);
        auto eq                 = body->add_instruction(migraphx::make_op("equal"), iter, l);
        auto beq                = body->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), eq);
        auto neq = body->add_instruction(migraphx::make_op("not"), beq);
        body->add_return({neq, val, val});

        auto rl = mm->add_instruction(
            migraphx::make_op("loop", {{"max_iterations", 8}}), {in_iter, in_cond, in_val}, {body});
        auto r0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), rl);
        auto r1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), rl);
        mm->add_return({r0, r1});

        return p;
    }
};
