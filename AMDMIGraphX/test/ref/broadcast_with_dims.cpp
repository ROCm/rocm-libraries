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
#include <migraphx/float_equal.hpp>

#include "test.hpp"

TEST_CASE(broadcast_with_dims_static0)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape input_shape{migraphx::shape::int32_type, {2}};
    migraphx::shape dims_shape{migraphx::shape::int64_type, {2}};
    auto input_param = mm->add_parameter("x", input_shape);
    auto dims_param  = mm->add_parameter("dims", dims_shape);
    mm->add_instruction(migraphx::make_op("broadcast_with_dims"), input_param, dims_param);
    p.compile(migraphx::make_target("ref"));

    std::vector<int32_t> input_data{-3, 3};
    std::vector<int64_t> dims_data{2, 1};
    migraphx::parameter_map params;
    params["x"]    = migraphx::argument(input_shape, input_data.data());
    params["dims"] = migraphx::argument(dims_shape, dims_data.data());
    auto result    = p.eval(params).back();
    auto output    = result.get<int32_t>();
    EXPECT(output.get_shape().lens() == std::vector<std::size_t>{2, 2});
    EXPECT(output.get_shape().strides() == std::vector<std::size_t>{0, 1});
    EXPECT(output(0, 0) == -3);
    EXPECT(output(0, 1) == 3);
    EXPECT(output(1, 0) == -3);
    EXPECT(output(1, 1) == 3);
}

TEST_CASE(broadcast_with_dims_static1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape input_shape{migraphx::shape::float_type, {2, 1}, {1, 0}};
    migraphx::shape dims_shape{migraphx::shape::int64_type, {1}};
    auto input_param = mm->add_parameter("x", input_shape);
    auto dims_param  = mm->add_parameter("dims", dims_shape);
    mm->add_instruction(migraphx::make_op("broadcast_with_dims"), input_param, dims_param);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{7, 11};
    std::vector<int64_t> dims_data{3};
    migraphx::parameter_map params;
    params["x"]    = migraphx::argument(input_shape, input_data.data());
    params["dims"] = migraphx::argument(dims_shape, dims_data.data());
    auto result    = p.eval(params).back();
    auto output    = result.get<float>();
    EXPECT(output.get_shape().lens() == std::vector<std::size_t>{2, 3});
    EXPECT(output.get_shape().strides() == std::vector<std::size_t>{1, 0});
    EXPECT(migraphx::float_equal(output(0, 0), 7.f));
    EXPECT(migraphx::float_equal(output(0, 1), 7.f));
    EXPECT(migraphx::float_equal(output(0, 2), 7.f));
    EXPECT(migraphx::float_equal(output(1, 0), 11.f));
    EXPECT(migraphx::float_equal(output(1, 1), 11.f));
    EXPECT(migraphx::float_equal(output(1, 2), 11.f));
}

TEST_CASE(broadcast_with_dims_static2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape input_shape{migraphx::shape::float_type, {2, 3}, {3, 1}};
    migraphx::shape dims_shape{migraphx::shape::int64_type, {3}};
    auto input_param = mm->add_parameter("x", input_shape);
    auto dims_param  = mm->add_parameter("dims", dims_shape);
    mm->add_instruction(migraphx::make_op("broadcast_with_dims"), input_param, dims_param);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data(6);
    std::iota(input_data.begin(), input_data.end(), 0.0);
    std::vector<int64_t> dims_data{4, 2, 3};
    migraphx::parameter_map params;
    params["x"]    = migraphx::argument(input_shape, input_data.data());
    params["dims"] = migraphx::argument(dims_shape, dims_data.data());
    auto result    = p.eval(params).back();
    auto output    = result.get<float>();
    EXPECT(output.get_shape().lens() == std::vector<std::size_t>{4, 2, 3});
    EXPECT(output.get_shape().strides() == std::vector<std::size_t>{0, 3, 1});
    std::vector<int> results_vector;
    result.visit([&](auto x) { results_vector.assign(x.begin(), x.end()); });
    std::vector<float> gold = {
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
    };
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(broadcast_with_dims_dyn)
{
    migraphx::program p;
    auto* mm                                            = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dds = {{2, 4}};
    migraphx::shape input_shape{migraphx::shape::int32_type, dds};
    migraphx::shape dims_shape{migraphx::shape::int64_type, {2}};
    auto input_param = mm->add_parameter("x", input_shape);
    auto dims_param  = mm->add_parameter("dims", dims_shape);
    mm->add_instruction(migraphx::make_op("broadcast_with_dims"), input_param, dims_param);
    p.compile(migraphx::make_target("ref"));

    std::vector<int32_t> input_data{-3, 3};
    std::vector<int64_t> dims_data{2, 2};
    migraphx::shape input_static_shape{migraphx::shape::int32_type, {2}};
    migraphx::parameter_map params;
    params["x"]    = migraphx::argument(input_static_shape, input_data.data());
    params["dims"] = migraphx::argument(dims_shape, dims_data.data());
    auto result    = p.eval(params).back();
    auto output    = result.get<int32_t>();
    EXPECT(output.get_shape().lens() == std::vector<std::size_t>{2, 2});
    EXPECT(output.get_shape().strides() == std::vector<std::size_t>{0, 1});
    EXPECT(output(0, 0) == -3);
    EXPECT(output(0, 1) == 3);
    EXPECT(output(1, 0) == -3);
    EXPECT(output(1, 1) == 3);
}

TEST_CASE(broadcast_with_dims_mismatch)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape input_shape{migraphx::shape::float_type, {2, 3}};
    migraphx::shape dims_shape{migraphx::shape::int64_type, {1}};
    auto input_param = mm->add_parameter("x", input_shape);
    auto dims_param  = mm->add_parameter("dims", dims_shape);
    mm->add_instruction(migraphx::make_op("broadcast_with_dims"), input_param, dims_param);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{3, 9};
    std::vector<int64_t> dims_data{6};
    migraphx::parameter_map params;
    params["x"]    = migraphx::argument(input_shape, input_data.data());
    params["dims"] = migraphx::argument(dims_shape, dims_data.data());
    EXPECT(test::throws([&] { std::ignore = p.eval(params).back(); }));
}
