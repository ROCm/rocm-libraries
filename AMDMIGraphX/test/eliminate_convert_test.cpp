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
#include <migraphx/eliminate_convert.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::eliminate_convert{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(nop_convert)
{
    migraphx::module m0;
    {
        auto s = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
        auto x = m0.add_parameter("x", s);
        auto t = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            x);
        m0.add_return({t});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        auto s = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
        auto x = m1.add_parameter("x", s);
        m1.add_return({x});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(nested_convert0)
{
    migraphx::module m0;
    {
        auto s = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
        auto x = m0.add_parameter("x", s);
        auto a = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
            x);
        auto b = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            a);
        m0.add_return({b});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        auto s = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
        auto x = m1.add_parameter("x", s);
        m1.add_return({x});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(nested_convert1)
{
    migraphx::module m0;
    {
        auto s = migraphx::shape{migraphx::shape::half_type, {1, 2, 3}};
        auto x = m0.add_parameter("x", s);
        auto a = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
            x);
        auto b = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            a);
        m0.add_return({b});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        auto s = migraphx::shape{migraphx::shape::half_type, {1, 2, 3}};
        auto x = m1.add_parameter("x", s);
        auto a = m1.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            x);
        m1.add_return({a});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(nested3_convert0)
{
    migraphx::module m0;
    {
        auto s = migraphx::shape{migraphx::shape::half_type, {1, 2, 3}};
        auto x = m0.add_parameter("x", s);
        auto a = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
            x);
        auto b = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            a);
        auto c = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
            b);
        m0.add_return({c});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        auto s = migraphx::shape{migraphx::shape::half_type, {1, 2, 3}};
        auto x = m1.add_parameter("x", s);
        auto a = m1.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
            x);
        m1.add_return({a});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(nested3_convert1)
{
    migraphx::module m0;
    {
        auto s = migraphx::shape{migraphx::shape::half_type, {1, 2, 3}};
        auto x = m0.add_parameter("x", s);
        auto a = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
            x);
        auto b = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
            a);
        auto c = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            b);
        m0.add_return({c});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        auto s = migraphx::shape{migraphx::shape::half_type, {1, 2, 3}};
        auto x = m1.add_parameter("x", s);
        auto a = m1.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            x);
        m1.add_return({a});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(nested3_convert2)
{
    migraphx::module m0;
    {
        auto s = migraphx::shape{migraphx::shape::half_type, {1, 2, 3}};
        auto x = m0.add_parameter("x", s);
        auto a = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
            x);
        auto b = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
            a);
        auto c = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::half_type)}}),
            b);
        m0.add_return({c});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        auto s = migraphx::shape{migraphx::shape::half_type, {1, 2, 3}};
        auto x = m1.add_parameter("x", s);
        m1.add_return({x});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(nested3_nop_convert)
{
    migraphx::module m0;
    {
        auto s = migraphx::shape{migraphx::shape::half_type, {1, 2, 3}};
        auto x = m0.add_parameter("x", s);
        auto a = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
            x);
        auto b = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
            a);
        auto c = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::half_type)}}),
            b);
        m0.add_return({c});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        auto s = migraphx::shape{migraphx::shape::half_type, {1, 2, 3}};
        auto x = m1.add_parameter("x", s);
        m1.add_return({x});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(nested_branch_convert0)
{
    migraphx::module m0;
    {
        auto s = migraphx::shape{migraphx::shape::half_type, {1, 2, 3}};
        auto x = m0.add_parameter("x", s);
        auto a = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
            x);
        auto b = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::half_type)}}),
            a);
        auto c = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
            b);
        auto d = m0.add_instruction(migraphx::make_op("add"), a, c);
        m0.add_return({d});
    }
    run_pass(m0);

    // TODO: Less than optimal end result, would need to look horizontally to reduce to a single
    // convert
    migraphx::module m1;
    {
        auto s = migraphx::shape{migraphx::shape::half_type, {1, 2, 3}};
        auto x = m1.add_parameter("x", s);
        auto a = m1.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
            x);
        auto b = m1.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
            x);
        auto d = m1.add_instruction(migraphx::make_op("add"), a, b);
        m1.add_return({d});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(nested_branch_convert1)
{
    migraphx::module m0;
    {
        auto s            = migraphx::shape{migraphx::shape::half_type, {1, 2, 3}};
        auto x            = m0.add_parameter("x", s);
        auto dbl_convert0 = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
            x);
        auto half_convert0 = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::half_type)}}),
            dbl_convert0);
        auto dbl_convert1 = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
            half_convert0);
        auto cos_ins      = m0.add_instruction(migraphx::make_op("cos"), half_convert0);
        auto dbl_convert2 = m0.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
            cos_ins);
        auto add_ins = m0.add_instruction(migraphx::make_op("add"), dbl_convert2, dbl_convert1);
        m0.add_return({add_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        auto s            = migraphx::shape{migraphx::shape::half_type, {1, 2, 3}};
        auto x            = m1.add_parameter("x", s);
        auto dbl_convert0 = m1.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
            x);
        auto cos_ins      = m1.add_instruction(migraphx::make_op("cos"), x);
        auto dbl_convert1 = m1.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
            cos_ins);
        auto add_ins = m1.add_instruction(migraphx::make_op("add"), dbl_convert1, dbl_convert0);
        m1.add_return({add_ins});
    }
    EXPECT(m0 == m1);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
