/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include "test.hpp"

TEST_CASE(check_undefined)
{
    migraphx::module m;
    auto und = m.add_instruction(migraphx::make_op("undefined"));
    auto cov = m.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), und);
    auto abs = m.add_instruction(migraphx::make_op("abs"), cov);

    migraphx::shape xs{migraphx::shape::float_type, {2, 3}};
    std::vector<float> datax = {1, 2, 3, 4, 5, 6};

    auto lit = m.add_literal(migraphx::literal(xs, datax));
    auto mul = m.add_instruction(migraphx::make_op("mul"), lit, lit);

    EXPECT(und->is_undefined());
    EXPECT(cov->is_undefined());
    EXPECT(abs->is_undefined());
    EXPECT(not lit->is_undefined());
    EXPECT(not mul->is_undefined());
}

TEST_CASE(check_replace_shape)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {3, 2}};
    auto input  = m.add_parameter("x", s);
    auto reduce = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), input);
    auto abs    = m.add_instruction(migraphx::make_op("abs"), reduce);
    auto sin    = m.add_instruction(migraphx::make_op("sin"), reduce);
    auto add    = m.add_instruction(migraphx::make_op("add"), abs, sin);

    reduce->replace(migraphx::make_op("reduce_sum", {{"axes", {1}}}));

    migraphx::shape r{migraphx::shape::float_type, {3, 1}};
    EXPECT(reduce->get_shape() == r);
    EXPECT(abs->get_shape() == r);
    EXPECT(sin->get_shape() == r);
    EXPECT(add->get_shape() == r);
}

TEST_CASE(check_replace_dag)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {3, 2}};
    auto input  = m.add_parameter("x", s);
    auto reduce = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), input);
    auto abs    = m.add_instruction(migraphx::make_op("abs"), reduce);
    auto sin    = m.add_instruction(migraphx::make_op("sin"), reduce);
    auto add    = m.add_instruction(migraphx::make_op("add"), abs, sin);
    auto add2   = m.add_instruction(migraphx::make_op("add"), add, reduce);

    reduce->replace(migraphx::make_op("reduce_sum", {{"axes", {1}}}));

    migraphx::shape r{migraphx::shape::float_type, {3, 1}};
    EXPECT(reduce->get_shape() == r);
    EXPECT(abs->get_shape() == r);
    EXPECT(sin->get_shape() == r);
    EXPECT(add->get_shape() == r);
    EXPECT(add2->get_shape() == r);
}
int main(int argc, const char* argv[]) { test::run(argc, argv); }
