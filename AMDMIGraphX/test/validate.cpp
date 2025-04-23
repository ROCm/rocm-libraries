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
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <basic_ops.hpp>
#include <test.hpp>
#include <rob.hpp>

TEST_CASE(simple_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(migraphx::make_op("add"), one, two);
    EXPECT(bool{mm->validate() == mm->end()});
    auto result = p.eval({});
    EXPECT(result.back() == migraphx::literal{3});
    EXPECT(result.back() != migraphx::literal{4});
}

TEST_CASE(out_of_order)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto ins = mm->add_instruction(migraphx::make_op("add"), one, two);
    mm->move_instruction(two, mm->end());
    EXPECT(bool{p.validate() == ins});
}

TEST_CASE(incomplete_args)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto ins = mm->add_instruction(migraphx::make_op("add"), one, two);
    ins->clear_arguments();
    EXPECT(bool{p.validate() == ins});
}

MIGRAPHX_ROB(access_ins_arguments,
             std::vector<migraphx::instruction_ref>,
             migraphx::instruction,
             arguments)

TEST_CASE(invalid_args)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto ins = mm->add_instruction(migraphx::make_op("add"), one, two);
    access_ins_arguments(*ins).clear();
    EXPECT(bool{mm->validate() == mm->begin()});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
