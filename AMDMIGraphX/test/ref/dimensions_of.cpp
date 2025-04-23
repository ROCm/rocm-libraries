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

TEST_CASE(dimensions_of_test0)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4, {2, 4}}, {3, 3}, {4, 4}}};
    auto p1 = mm->add_parameter("x", s);
    mm->add_instruction(migraphx::make_op("dimensions_of", {{"end", 3}}), p1);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> x_data(24, 1.0);
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(input_fixed_shape, x_data.data());
    auto result = p.eval(params).back();
    std::vector<int64_t> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<int64_t> gold = {2, 3, 4};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(dimensions_of_test1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4, {1, 4}}, {3, 3}, {3, 8}, {3, 8}}};
    auto p1 = mm->add_parameter("x", s);
    mm->add_instruction(migraphx::make_op("dimensions_of", {{"start", 2}, {"end", 4}}), p1);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> x_data(48, 1.0);
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 3, 4, 4}};
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(input_fixed_shape, x_data.data());
    auto result = p.eval(params).back();
    std::vector<int64_t> results_vector(2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<int64_t> gold = {4, 4};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
