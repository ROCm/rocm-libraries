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

TEST_CASE(contiguous_test)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 3, 2, 2}, {12, 1, 6, 3}};
    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0);

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(migraphx::literal{a_shape, data});
    mm->add_instruction(migraphx::make_op("contiguous"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<size_t> new_strides = {12, 4, 2, 1};
    EXPECT(result.get_shape().strides() == new_strides);

    std::vector<float> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(contiguous_param_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 3, 2, 2}, {12, 1, 6, 3}};
    auto a = mm->add_parameter("X", a_shape);
    mm->add_instruction(migraphx::make_op("contiguous"), a);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0);
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(a_shape, data.data());
    auto result = p.eval(params).back();

    std::vector<size_t> new_strides = {12, 4, 2, 1};
    EXPECT(result.get_shape().strides() == new_strides);

    std::vector<float> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(contiguous_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape dyn_shape{migraphx::shape::float_type, {{1, 1}, {2, 6}, {2, 2}, {2, 2}}};
    auto input = mm->add_parameter("X", dyn_shape);
    mm->add_instruction(migraphx::make_op("contiguous"), input);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape static_shape{migraphx::shape::float_type, {1, 3, 2, 2}, {12, 1, 6, 3}};
    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0);
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(static_shape, data.data());
    auto result = p.eval(params).back();

    std::vector<size_t> new_strides = {12, 4, 2, 1};
    EXPECT(result.get_shape().strides() == new_strides);

    std::vector<float> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
