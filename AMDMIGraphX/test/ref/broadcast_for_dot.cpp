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

#include "test.hpp"

TEST_CASE(broadcast_for_dot_static)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::float_type, {2, 4}};
    std::vector<float> data0(8);
    std::iota(data0.begin(), data0.end(), 0.0);
    std::vector<float> data1(16);
    std::iota(data1.begin(), data1.end(), 9.0);
    migraphx::shape s1{migraphx::shape::float_type, {1, 2, 4, 2}};
    auto l0                = mm->add_literal(migraphx::literal{s0, data0});
    auto l1                = mm->add_literal(migraphx::literal{s1, data1});
    auto broadcast_for_dot = mm->add_instruction(migraphx::make_op("broadcast_for_dot"), l0, l1);
    mm->add_return({broadcast_for_dot});
    p.compile(migraphx::make_target("ref"));

    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold(16);
    std::iota(gold.begin(), gold.begin() + 8, 0.0);
    std::iota(gold.begin() + 8, gold.end(), 0.0);
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(broadcast_for_dot_dyn)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::int32_type, {{2, 6}, {4, 4}}};
    migraphx::shape s1{migraphx::shape::int32_type, {{1, 4}, {2, 2}, {4, 4}, {4, 6}}};
    auto p0                = mm->add_parameter("0", s0);
    auto p1                = mm->add_parameter("1", s1);
    auto broadcast_for_dot = mm->add_instruction(migraphx::make_op("broadcast_for_dot"), p0, p1);
    mm->add_return({broadcast_for_dot});
    p.compile(migraphx::make_target("ref"));

    std::vector<int> data0(8);
    std::iota(data0.begin(), data0.end(), 0);
    std::vector<int> data1(16);
    std::iota(data1.begin(), data1.end(), 9);
    migraphx::shape fixed_shape0{migraphx::shape::int32_type, {2, 4}};
    migraphx::shape fixed_shape1{migraphx::shape::int32_type, {1, 2, 4, 2}};
    migraphx::parameter_map params;
    params["0"] = migraphx::argument(fixed_shape0, data0.data());
    params["1"] = migraphx::argument(fixed_shape1, data1.data());
    auto result = p.eval(params).back();
    std::vector<int> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<int> gold(16);
    std::iota(gold.begin(), gold.begin() + 8, 0.0);
    std::iota(gold.begin() + 8, gold.end(), 0.0);
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
