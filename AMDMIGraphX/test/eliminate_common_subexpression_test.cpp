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
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(
        p, {migraphx::eliminate_common_subexpression{}, migraphx::dead_code_elimination{}});
}

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(
        m, {migraphx::eliminate_common_subexpression{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(cse_test1)
{
    migraphx::module m1;
    {
        auto one  = m1.add_literal(1);
        auto two  = m1.add_literal(2);
        auto sum1 = m1.add_instruction(migraphx::make_op("add"), one, two);
        auto sum2 = m1.add_instruction(migraphx::make_op("add"), one, two);
        auto sum3 = m1.add_instruction(migraphx::make_op("add"), sum1, sum2);
        m1.add_instruction(pass_op{}, sum3);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto one  = m2.add_literal(1);
        auto two  = m2.add_literal(2);
        auto sum1 = m2.add_instruction(migraphx::make_op("add"), one, two);
        auto sum3 = m2.add_instruction(migraphx::make_op("add"), sum1, sum1);
        m2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(cse_test2)
{
    migraphx::module m1;
    {
        auto one  = m1.add_literal(1);
        auto two  = m1.add_literal(2);
        auto sum1 = m1.add_instruction(migraphx::make_op("add"), one, two);
        auto sum2 = m1.add_instruction(migraphx::make_op("add"), two, one);
        auto sum3 = m1.add_instruction(migraphx::make_op("add"), sum1, sum2);
        m1.add_instruction(pass_op{}, sum3);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto one  = m2.add_literal(1);
        auto two  = m2.add_literal(2);
        auto sum1 = m2.add_instruction(migraphx::make_op("add"), one, two);
        auto sum2 = m2.add_instruction(migraphx::make_op("add"), two, one);
        auto sum3 = m2.add_instruction(migraphx::make_op("add"), sum1, sum2);
        m2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(cse_test3)
{
    migraphx::module m1;
    {
        auto one  = m1.add_literal(1);
        auto two  = m1.add_literal(1);
        auto sum1 = m1.add_instruction(migraphx::make_op("add"), one, two);
        auto sum2 = m1.add_instruction(migraphx::make_op("add"), two, one);
        auto sum3 = m1.add_instruction(migraphx::make_op("add"), sum1, sum2);
        m1.add_instruction(pass_op{}, sum3);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto one  = m2.add_literal(1);
        auto sum1 = m2.add_instruction(migraphx::make_op("add"), one, one);
        auto sum3 = m2.add_instruction(migraphx::make_op("add"), sum1, sum1);
        m2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(cse_test4)
{
    migraphx::module m1;
    {
        auto one  = m1.add_literal(1);
        auto two  = m1.add_literal(1);
        auto sum1 = m1.add_instruction(migraphx::make_op("add"), one, two);
        auto sum2 = m1.add_instruction(migraphx::make_op("add"), two, one);
        auto sum3 = m1.add_instruction(migraphx::make_op("add"), sum1, one);
        auto sum4 = m1.add_instruction(migraphx::make_op("add"), sum2, two);
        auto sum5 = m1.add_instruction(migraphx::make_op("add"), sum4, sum3);
        m1.add_instruction(pass_op{}, sum5);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto one  = m2.add_literal(1);
        auto sum1 = m2.add_instruction(migraphx::make_op("add"), one, one);
        auto sum3 = m2.add_instruction(migraphx::make_op("add"), sum1, one);
        auto sum5 = m2.add_instruction(migraphx::make_op("add"), sum3, sum3);
        m2.add_instruction(pass_op{}, sum5);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(cse_test_literal)
{
    migraphx::module m1;
    {
        auto six1  = m1.add_literal(6);
        auto zero1 = m1.add_literal(0);
        auto six2  = m1.add_literal(6);
        auto zero2 = m1.add_literal(0);
        auto six3  = m1.add_literal(6);
        auto zero3 = m1.add_literal(0);

        auto sum1 = m1.add_instruction(migraphx::make_op("add"), six1, zero1);
        auto sum2 = m1.add_instruction(migraphx::make_op("add"), six2, zero2);
        auto sum3 = m1.add_instruction(migraphx::make_op("add"), six3, zero3);
        auto sum4 = m1.add_instruction(migraphx::make_op("add"), sum1, sum2);
        auto sum5 = m1.add_instruction(migraphx::make_op("add"), sum3, sum4);
        m1.add_instruction(pass_op{}, sum5);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto six  = m2.add_literal(6);
        auto zero = m2.add_literal(0);
        auto sum1 = m2.add_instruction(migraphx::make_op("add"), six, zero);
        auto sum2 = m2.add_instruction(migraphx::make_op("add"), sum1, sum1);
        auto sum3 = m2.add_instruction(migraphx::make_op("add"), sum1, sum2);
        m2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(cse_test_submodule)
{
    migraphx::shape si{migraphx::shape::int64_type};
    migraphx::shape s{migraphx::shape::int64_type, {1}};
    migraphx::shape sc{migraphx::shape::bool_type};

    auto create_program = [&](bool remove_literal = false) {
        migraphx::program p;
        std::vector<bool> vc    = {true};
        std::vector<int64_t> vd = {3};
        auto* mm                = p.get_main_module();

        auto in_cond = mm->add_parameter("ccond", sc);
        auto in_val  = mm->add_parameter("val", s);
        auto b0      = mm->add_literal(migraphx::literal(sc, vc));
        auto b1      = b0;
        if(not(remove_literal))
            b1 = mm->add_literal(migraphx::literal(sc, vc));

        auto* body1 = p.create_module("loop_module1");
        body1->add_parameter("#loop_module_in_1", sc);
        auto in_v1 = body1->add_parameter("#loop_module_in_2", s);
        auto l1    = body1->add_literal(migraphx::literal(si, vd));
        auto ad1   = body1->add_instruction(migraphx::make_op("add"), l1, l1);
        auto val1  = body1->add_instruction(migraphx::make_op("add"), in_v1, ad1);
        auto cond1 = body1->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), b0);
        auto cond2 = body1->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), b1);
        body1->add_return({cond1, cond2, val1, val1});

        auto* body2 = p.create_module("loop_module2");
        body2->add_parameter("#loop_module_in_1", sc);
        auto in_v2 = body2->add_parameter("#loop_module_in_2", s);
        auto l2    = body2->add_literal(migraphx::literal(si, vd));
        auto ad2   = body2->add_instruction(migraphx::make_op("add"), l2, l2);
        auto val2  = body2->add_instruction(migraphx::make_op("add"), in_v2, ad2);
        auto cond3 = body2->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), b1);
        body2->add_return({cond3, val2, val2});

        auto loop1 = mm->add_instruction(
            migraphx::make_op("loop", {{"max_iterations", 1}}), {in_cond, in_val}, {body1});
        auto loop2 = mm->add_instruction(
            migraphx::make_op("loop", {{"max_iterations", 1}}), {in_cond, in_val}, {body2});

        mm->add_return({loop1, loop2});

        return p;
    };
    auto p = create_program();
    run_pass(p);
    EXPECT(p == create_program(true));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
