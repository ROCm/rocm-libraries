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

TEST_CASE(convert_downcast_overflow_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    std::vector<float> data(4, 2 * std::numeric_limits<migraphx::half>::max());
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
                        l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<migraphx::half> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(std::all_of(results_vector.begin(), results_vector.end(), [](const auto& x) {
        return x == std::numeric_limits<migraphx::half>::max();
    }));
}

TEST_CASE(convert_downcast_underflow_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    std::vector<float> data(4, 2 * std::numeric_limits<migraphx::half>::lowest());
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
                        l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<migraphx::half> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(std::all_of(results_vector.begin(), results_vector.end(), [](const auto& x) {
        return x == std::numeric_limits<migraphx::half>::lowest();
    }));
}

TEST_CASE(convert_nan_upcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::half_type, {2, 2}};
    std::vector<migraphx::half> data(4, std::numeric_limits<migraphx::half>::quiet_NaN());
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4, -1);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(std::all_of(
        results_vector.begin(), results_vector.end(), [](const auto& x) { return std::isnan(x); }));
}

TEST_CASE(convert_nan_downcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::double_type, {2, 2}};
    std::vector<double> data(4, std::numeric_limits<double>::quiet_NaN());
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4, -1);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(std::all_of(
        results_vector.begin(), results_vector.end(), [](const auto& x) { return std::isnan(x); }));
}

TEST_CASE(convert_nan_double_convert_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::double_type, {2, 2}};
    std::vector<double> data(4, std::numeric_limits<double>::quiet_NaN());
    auto l   = mm->add_literal(migraphx::literal{s, data});
    auto f_l = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), l);
    mm->add_instruction(migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
                        f_l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<migraphx::half> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(std::all_of(
        results_vector.begin(), results_vector.end(), [](const auto& x) { return std::isnan(x); }));
}

TEST_CASE(convert_nan_convert_updown_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    std::vector<float> data(4, std::numeric_limits<float>::quiet_NaN());
    auto l   = mm->add_literal(migraphx::literal{s, data});
    auto f_l = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), l);
    auto h_l = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), f_l);
    mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), h_l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(std::all_of(
        results_vector.begin(), results_vector.end(), [](const auto& x) { return std::isnan(x); }));
}
