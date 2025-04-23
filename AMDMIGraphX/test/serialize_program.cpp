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
#include <migraphx/register_target.hpp>
#include <migraphx/load_save.hpp>
#include "test.hpp"
#include <migraphx/make_op.hpp>

#include <cstdio>

static migraphx::program create_program()
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x   = mm->add_parameter("x", {migraphx::shape::int32_type});
    auto two = mm->add_literal(2);
    auto add = mm->add_instruction(migraphx::make_op("add"), x, two);
    mm->add_return({add});
    return p;
}

TEST_CASE(as_value)
{
    migraphx::program p1 = create_program();
    migraphx::program p2;
    p2.from_value(p1.to_value());
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(as_msgpack)
{
    migraphx::file_options options;
    options.format           = "msgpack";
    migraphx::program p1     = create_program();
    std::vector<char> buffer = migraphx::save_buffer(p1, options);
    migraphx::program p2     = migraphx::load_buffer(buffer, options);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(as_json)
{
    migraphx::file_options options;
    options.format           = "json";
    migraphx::program p1     = create_program();
    std::vector<char> buffer = migraphx::save_buffer(p1, options);
    migraphx::program p2     = migraphx::load_buffer(buffer, options);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(as_file)
{
    std::string filename = "migraphx_program.mxr";
    migraphx::program p1 = create_program();
    migraphx::save(p1, filename);
    migraphx::program p2 = migraphx::load(filename);
    std::remove(filename.c_str());
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(compiled)
{
    migraphx::program p1 = create_program();
    p1.compile(migraphx::make_target("ref"));
    std::vector<char> buffer = migraphx::save_buffer(p1);
    migraphx::program p2     = migraphx::load_buffer(buffer);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(unknown_format)
{
    migraphx::file_options options;
    options.format = "???";

    EXPECT(test::throws([&] { migraphx::save_buffer(create_program(), options); }));
    EXPECT(test::throws([&] { migraphx::load_buffer(std::vector<char>{}, options); }));
}

TEST_CASE(program_with_module)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sd{migraphx::shape::float_type, {2, 3}};
    auto x = mm->add_parameter("x", sd);

    std::vector<float> one(sd.elements(), 1);
    std::vector<float> two(sd.elements(), 2);

    auto* then_smod = p.create_module("then_smod");
    auto l1         = then_smod->add_literal(migraphx::literal{sd, one});
    auto r1         = then_smod->add_instruction(migraphx::make_op("add"), x, l1);
    then_smod->add_return({r1});

    auto* else_smod = p.create_module("else_smod");
    auto l2         = else_smod->add_literal(migraphx::literal{sd, two});
    auto r2         = else_smod->add_instruction(migraphx::make_op("mul"), x, l2);
    else_smod->add_return({r2});

    migraphx::shape s_cond{migraphx::shape::bool_type, {1}};
    auto cond = mm->add_parameter("cond", s_cond);
    auto ret  = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_smod, else_smod});
    mm->add_return({ret});

    migraphx::program p1 = p;
    auto v               = p.to_value();
    auto v1              = p1.to_value();
    EXPECT(v == v1);

    std::stringstream ss;
    p.print_cpp(ss);
    std::stringstream ss1;
    p1.print_cpp(ss1);
    EXPECT(ss.str() == ss1.str());

    migraphx::program p2;
    p2.from_value(v);
    EXPECT(p1.sort() == p2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
