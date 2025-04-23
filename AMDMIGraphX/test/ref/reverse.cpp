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

TEST_CASE(reverse_test_axis0)
{
    migraphx::shape in_shape{migraphx::shape::float_type, {2, 16}};
    std::vector<float> data(32);
    std::iota(data.begin(), data.end(), 1);
    migraphx::program p;
    auto* mm              = p.get_main_module();
    auto l                = mm->add_literal(migraphx::literal{in_shape, data});
    std::vector<int> axes = {0};
    mm->add_instruction(migraphx::make_op("reverse", {{"axes", axes}}), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::swap_ranges(gold.begin(), gold.begin() + 16, gold.begin() + 16);
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(reverse_test_axis1)
{
    migraphx::shape in_shape{migraphx::shape::float_type, {2, 16}};
    std::vector<float> data(32);
    std::iota(data.begin(), data.end(), 1);
    migraphx::program p;
    auto* mm              = p.get_main_module();
    auto l                = mm->add_literal(migraphx::literal{in_shape, data});
    std::vector<int> axes = {1};
    mm->add_instruction(migraphx::make_op("reverse", {{"axes", axes}}), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::reverse(gold.begin(), gold.begin() + 16);
    std::reverse(gold.end() - 16, gold.end());
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(reverse_test_axis10)
{
    migraphx::shape in_shape{migraphx::shape::float_type, {2, 16}};
    std::vector<float> data(32);
    std::iota(data.begin(), data.end(), 1);
    migraphx::program p;
    auto* mm              = p.get_main_module();
    auto l                = mm->add_literal(migraphx::literal{in_shape, data});
    std::vector<int> axes = {1, 0};
    mm->add_instruction(migraphx::make_op("reverse", {{"axes", axes}}), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::reverse(gold.begin(), gold.begin() + 16);
    std::reverse(gold.end() - 16, gold.end());
    std::swap_ranges(gold.begin(), gold.begin() + 16, gold.begin() + 16);
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
