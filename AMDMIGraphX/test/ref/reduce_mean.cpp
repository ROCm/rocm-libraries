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

TEST_CASE(reduce_mean_axis02)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {0, 2}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{5.5, 7.5};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_mean_variable_axes02)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape x_shape{migraphx::shape::float_type, {3, 2, 2}};
    auto x = mm->add_parameter("x", x_shape);
    migraphx::shape axes_shape{migraphx::shape::int64_type, {2}};
    auto axes = mm->add_parameter("axes", axes_shape);
    mm->add_instruction(migraphx::make_op("reduce_mean"), x, axes);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;
    std::vector<float> x_arg{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    pm["x"] = migraphx::argument(x_shape, x_arg.data());
    std::vector<int64_t> axes_arg{0, 2};
    pm["axes"]  = migraphx::argument(axes_shape, axes_arg.data());
    auto result = p.eval(pm).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold{5.5, 7.5};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_mean_axis1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {1}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{2, 3, 6, 7, 10, 11};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_mean_variable_axis1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape x_shape{migraphx::shape::float_type, {3, 2, 2}};
    auto x = mm->add_parameter("x", x_shape);
    migraphx::shape axes_shape{migraphx::shape::int64_type, {1}};
    auto axes = mm->add_parameter("axes", axes_shape);
    mm->add_instruction(migraphx::make_op("reduce_mean"), x, axes);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;
    std::vector<float> x_arg{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    pm["x"] = migraphx::argument(x_shape, x_arg.data());
    std::vector<int64_t> axes_arg{1};
    pm["axes"]  = migraphx::argument(axes_shape, axes_arg.data());
    auto result = p.eval(pm).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold{2, 3, 6, 7, 10, 11};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_mean_axis12)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {1, 2}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{2.5f, 6.5f, 10.5f};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_mean_variable_axes12)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape x_shape{migraphx::shape::float_type, {3, 2, 2}};
    auto x = mm->add_parameter("x", x_shape);
    migraphx::shape axes_shape{migraphx::shape::int64_type, {2}};
    auto axes = mm->add_parameter("axes", axes_shape);
    mm->add_instruction(migraphx::make_op("reduce_mean"), x, axes);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;
    std::vector<float> x_arg{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    pm["x"] = migraphx::argument(x_shape, x_arg.data());
    std::vector<int64_t> axes_arg{1, 2};
    pm["axes"]  = migraphx::argument(axes_shape, axes_arg.data());
    auto result = p.eval(pm).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold{2.5, 6.5, 10.5};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_mean_axis2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_mean_variable_axis2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape x_shape{migraphx::shape::float_type, {3, 2, 2}};
    auto x = mm->add_parameter("x", x_shape);
    migraphx::shape axes_shape{migraphx::shape::int64_type, {1}};
    auto axes = mm->add_parameter("axes", axes_shape);
    mm->add_instruction(migraphx::make_op("reduce_mean"), x, axes);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;
    std::vector<float> x_arg{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    pm["x"] = migraphx::argument(x_shape, x_arg.data());
    std::vector<int64_t> axes_arg{2};
    pm["axes"]  = migraphx::argument(axes_shape, axes_arg.data());
    auto result = p.eval(pm).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold{1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_mean_int)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {1, 2}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<int> gold{2, 6, 10};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_mean_variable_axes12_int)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape x_shape{migraphx::shape::int32_type, {3, 2, 2}};
    auto x = mm->add_parameter("x", x_shape);
    migraphx::shape axes_shape{migraphx::shape::int64_type, {2}};
    auto axes = mm->add_parameter("axes", axes_shape);
    mm->add_instruction(migraphx::make_op("reduce_mean"), x, axes);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;
    std::vector<int32_t> x_arg{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    pm["x"] = migraphx::argument(x_shape, x_arg.data());
    std::vector<int64_t> axes_arg{1, 2};
    pm["axes"]  = migraphx::argument(axes_shape, axes_arg.data());
    auto result = p.eval(pm).back();
    std::vector<int32_t> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<int32_t> gold{2, 6, 10};
    EXPECT(results_vector == gold);
}
