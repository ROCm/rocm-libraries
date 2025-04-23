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
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(pointwise_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l1  = mm->add_literal(migraphx::literal{s, {-1, 0, 1}});
    auto l2  = mm->add_literal(migraphx::literal{s, {1, 2, 3}});
    auto* pm = p.create_module("pointwise");
    {
        auto x1  = pm->add_parameter("x1", {migraphx::shape::float_type});
        auto x2  = pm->add_parameter("x2", {migraphx::shape::float_type});
        auto add = pm->add_instruction(migraphx::make_op("add"), x1, x2);
        pm->add_return({add});
    }
    mm->add_instruction(migraphx::make_op("pointwise"), {l1, l2}, {pm});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0, 2, 4};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(pointwise_multi_out_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto a1  = mm->add_literal(migraphx::literal{s, {-1, 0, 1}});
    auto a2  = mm->add_literal(migraphx::literal{s, {1, 16, 3}});
    auto* pm = p.create_module("pointwise");
    {
        auto x1   = pm->add_parameter("x1", {migraphx::shape::float_type});
        auto x2   = pm->add_parameter("x2", {migraphx::shape::float_type});
        auto add  = pm->add_instruction(migraphx::make_op("add"), x1, x2);
        auto sqrt = pm->add_instruction(migraphx::make_op("sqrt"), add);
        pm->add_return({add, sqrt});
    }
    mm->add_instruction(migraphx::make_op("pointwise"), {a1, a2}, {pm});
    p.compile(migraphx::make_target("ref"));
    auto results = p.eval({}).back().get_sub_objects();

    std::vector<float> gold1 = {0, 16, 4};
    std::vector<float> gold2 = {0, 4, 2};
    EXPECT(results[0].to_vector<float>() == gold1);
    EXPECT(results[1].to_vector<float>() == gold2);
}
