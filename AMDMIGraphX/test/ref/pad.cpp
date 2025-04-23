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

TEST_CASE(pad_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    mm->add_instruction(migraphx::make_op("pad", {{"pads", {1, 1, 1, 1}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(pad_test_asym)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    mm->add_instruction(migraphx::make_op("pad", {{"pads", {0, 0, 1, 1}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(9);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1, 2, 0, 3, 4, 0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(pad_test_highest_half)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::half_type, {2, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    mm->add_instruction(
        migraphx::make_op("pad",
                          {{"pads", {1, 1, 1, 1}}, {"value", std::numeric_limits<float>::max()}}),
        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    const float x = std::numeric_limits<migraphx::half>::max();
    std::vector<float> gold{x, x, x, x, x, 1, 2, x, x, 3, 4, x, x, x, x, x};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(pad_test_lowest_half)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::half_type, {2, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    mm->add_instruction(
        migraphx::make_op(
            "pad", {{"pads", {1, 1, 1, 1}}, {"value", std::numeric_limits<float>::lowest()}}),
        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    const float x = std::numeric_limits<migraphx::half>::lowest();
    std::vector<float> gold{x, x, x, x, x, 1, 2, x, x, 3, 4, x, x, x, x, x};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(pad_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{2, 4, {2}}, {2, 4, {2}}}};
    auto x = mm->add_parameter("x", s);
    mm->add_instruction(migraphx::make_op("pad", {{"pads", {1, 1, 1, 1}}}), x);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data = {1, 2, 3, 4};
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {2, 2}};
    params["x"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
