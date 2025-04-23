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

#include <migraphx/program.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/register_target.hpp>
#include <sstream>
#include <migraphx/apply_alpha_beta.hpp>
#include "test.hpp"
#include <migraphx/make_op.hpp>

#include <basic_ops.hpp>

static migraphx::program create_program()
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x = mm->add_parameter("x", {migraphx::shape::int64_type});
    auto y = mm->add_parameter("y", {migraphx::shape::int64_type});

    auto sum = mm->add_instruction(sum_op{}, x, y);
    auto one = mm->add_literal(1);
    mm->add_instruction(sum_op{}, sum, one);

    return p;
}

TEST_CASE(program_equality)
{
    migraphx::program x = create_program();
    migraphx::program y = create_program();

    EXPECT(x.size() == 1);
    EXPECT(x == y);
}

TEST_CASE(program_not_equality1)
{
    migraphx::program x;
    migraphx::program y = create_program();
    EXPECT(x != y);
    x = y;
    EXPECT(x == y);
}

TEST_CASE(program_not_equality2)
{
    migraphx::program x;
    migraphx::program y = create_program();
    EXPECT(x != y);
    y = x;
    EXPECT(x == y);
}

TEST_CASE(program_default_copy_construct)
{
    migraphx::program x;
    migraphx::program y;
    EXPECT(x == y);
}

TEST_CASE(program_print)
{
    migraphx::program p = create_program();
    auto* mm            = p.get_main_module();
    auto in1            = mm->end();

    // print end instruction
    p.debug_print(in1);

    // print instruction not in the program
    auto p2   = p;
    auto* mm2 = p2.get_main_module();
    auto in2  = mm2->begin();
    p.debug_print(in2);

    // print last instruction
    auto in3 = std::prev(in1);
    p.debug_print(in3);
}

TEST_CASE(program_annotate)
{
    migraphx::program p1 = create_program();
    migraphx::program p2 = create_program();

    std::stringstream ss1;
    p1.annotate(ss1, [](auto ins) { std::cout << ins->name() << "_1" << std::endl; });

    std::stringstream ss2;
    p2.annotate(ss2, [](auto ins) { std::cout << ins->name() << "_1" << std::endl; });

    EXPECT(ss1.str() == ss2.str());
}

TEST_CASE(program_copy)
{
    auto create_program_1 = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape s{migraphx::shape::float_type, {3, 4, 5}};
        std::vector<float> data(3 * 4 * 5);
        std::iota(data.begin(), data.end(), 1.0f);
        auto l2  = mm->add_literal(migraphx::literal(s, data));
        auto p1  = mm->add_parameter("x", s);
        auto po  = mm->add_outline(s);
        auto sum = mm->add_instruction(migraphx::make_op("add"), l2, po);
        mm->add_instruction(migraphx::make_op("mul"), sum, p1);

        return p;
    };

    {
        auto p1 = create_program_1();
        migraphx::program p2{};
        p2 = p1;

        p2.compile(migraphx::make_target("ref"));
        EXPECT(p1 != p2);

        p1.compile(migraphx::make_target("ref"));
        EXPECT(p1 == p2);

        EXPECT(p1.get_parameter_names() == p2.get_parameter_names());
    }

    {
        auto p1 = create_program_1();
        auto p2(p1);
        EXPECT(p1 == p2);

        p1.compile(migraphx::make_target("ref"));
        EXPECT(p1 != p2);

        p2 = p1;
        EXPECT(p1 == p2);
    }

    {
        auto p1 = create_program_1();
        auto p2 = create_program();
        EXPECT(p1 != p2);

        p2 = p1;
        EXPECT(p1 == p2);

        p1.compile(migraphx::make_target("ref"));
        p2.compile(migraphx::make_target("ref"));

        EXPECT(p1 == p2);
    }

    {
        migraphx::program p1;
        auto* mm1 = p1.get_main_module();

        migraphx::shape s1{migraphx::shape::float_type, {2, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {3, 6}};
        migraphx::shape s3{migraphx::shape::float_type, {2, 6}};
        auto para1 = mm1->add_parameter("m1", s1);
        auto para2 = mm1->add_parameter("m2", s2);
        auto para3 = mm1->add_parameter("m3", s3);
        migraphx::add_apply_alpha_beta(
            *mm1, {para1, para2, para3}, migraphx::make_op("dot"), 0.31f, 0.28f);
        migraphx::program p2{};
        p2 = p1;
        EXPECT(p2 == p1);

        p1.compile(migraphx::make_target("ref"));
        p2.compile(migraphx::make_target("ref"));
        EXPECT(p2 == p1);
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
