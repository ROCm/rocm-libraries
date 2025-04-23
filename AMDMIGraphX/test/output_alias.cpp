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
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <test.hpp>
#include <basic_ops.hpp>

TEST_CASE(simple_alias)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(1);
    auto p1  = mm->add_instruction(pass_op{}, l);
    EXPECT(bool{migraphx::instruction::get_output_alias(l) == l});
    EXPECT(bool{migraphx::instruction::get_output_alias(p1) == l});
}

TEST_CASE(cascade_alias)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(1);
    auto p1  = mm->add_instruction(pass_op{}, l);
    auto p2  = mm->add_instruction(pass_op{}, p1);
    auto p3  = mm->add_instruction(pass_op{}, p2);
    EXPECT(bool{migraphx::instruction::get_output_alias(l) == l});
    EXPECT(bool{migraphx::instruction::get_output_alias(p1) == l});
    EXPECT(bool{migraphx::instruction::get_output_alias(p2) == l});
    EXPECT(bool{migraphx::instruction::get_output_alias(p3) == l});
}

TEST_CASE(no_alias)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(1);
    auto y   = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, x, y);
    EXPECT(bool{migraphx::instruction::get_output_alias(sum) == sum});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
