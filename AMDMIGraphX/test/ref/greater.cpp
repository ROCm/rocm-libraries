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
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(greater_brcst_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::float_type, {3, 3}};
    auto l0 =
        mm->add_literal(migraphx::literal{s0, {1.1, 1.5, 0.1, -1.1, -1.5, -0.6, 0.0, 2.0, -2.0}});
    migraphx::shape s1{migraphx::shape::float_type, {3, 1}};
    auto l1  = mm->add_literal(migraphx::literal{s1, {1.1, -1.5, 0.0}});
    auto bl1 = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 3}}}), l1);
    auto gr  = mm->add_instruction(migraphx::make_op("greater"), l0, bl1);
    auto r   = mm->add_instruction(
        migraphx::make_op("convert",
                            {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        gr);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<bool> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold = {false, true, false, true, false, true, false, true, false};
    EXPECT(results_vector == gold);
}

TEST_CASE(greater_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {9}};
    auto l0 =
        mm->add_literal(migraphx::literal{s, {1.1, 1.5, 0.1, -1.1, -1.5, -0.6, 0.0, 2.0, -2.0}});
    auto l1 =
        mm->add_literal(migraphx::literal{s, {1.1, 1.6, -0.1, -1.2, -1.5, -0.7, 0.0, 2.3, -2.1}});
    auto gr = mm->add_instruction(migraphx::make_op("greater"), l0, l1);
    auto r  = mm->add_instruction(
        migraphx::make_op("convert",
                           {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        gr);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<bool> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold = {false, false, true, true, false, true, false, false, true};
    EXPECT(results_vector == gold);
}

TEST_CASE(greater_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dd{{8, 10, {9}}};
    migraphx::shape s{migraphx::shape::float_type, dd};
    auto left  = mm->add_parameter("l", s);
    auto right = mm->add_parameter("r", s);
    auto gr    = mm->add_instruction(migraphx::make_op("greater"), left, right);
    auto r     = mm->add_instruction(
        migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        gr);
    mm->add_return({r});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> left_data{1.1, 1.5, 0.1, -1.1, -1.5, -0.6, 0.0, 2.0, -2.0};
    std::vector<float> right_data{1.1, 1.6, -0.1, -1.2, -1.5, -0.7, 0.0, 2.3, -2.1};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {9}};
    params0["l"] = migraphx::argument(input_fixed_shape0, left_data.data());
    params0["r"] = migraphx::argument(input_fixed_shape0, right_data.data());
    auto result  = p.eval(params0).back();
    std::vector<bool> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold = {false, false, true, true, false, true, false, false, true};
    EXPECT(results_vector == gold);
}
