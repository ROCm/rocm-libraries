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
#include <migraphx/onnx.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(fill_static_int)
{
    // Note this case can be simplified to a literal
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape lit_shape{migraphx::shape::int64_type, {1}, {0}};
    std::vector<int64_t> lit_data = {3};
    auto l                        = mm->add_literal(migraphx::literal{lit_shape, lit_data});
    migraphx::shape data_shape{migraphx::shape::int64_type, {3, 4, 4}};
    auto input = mm->add_parameter("x", data_shape);
    mm->add_instruction(migraphx::make_op("fill"), l, input);
    p.compile(migraphx::make_target("ref"));

    std::vector<int64_t> input_data(48);
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(data_shape, input_data.data());
    auto result = p.eval(params).back();
    std::vector<int64_t> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<int64_t> gold(48, 3);
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(fill_dyn_float)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape lit_shape{migraphx::shape::float_type, {1}, {0}};
    std::vector<float> lit_data = {7.36};
    auto l                      = mm->add_literal(migraphx::literal{lit_shape, lit_data});
    migraphx::shape data_shape{migraphx::shape::float_type,
                               {{1, 4}, {4, 8, {4, 6, 8}}, {4, 8, {4, 6, 8}}}};
    auto input = mm->add_parameter("x", data_shape);
    mm->add_instruction(migraphx::make_op("fill"), l, input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data(72);
    migraphx::parameter_map params;
    migraphx::shape static_shape = {migraphx::shape::float_type, {2, 6, 6}};
    params["x"]                  = migraphx::argument(static_shape, input_data.data());
    auto result                  = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold(72, 7.36);
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(fill_var_default_value)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape dv_shape{migraphx::shape::int64_type, {1}, {0}};
    auto dv = mm->add_parameter("dv", dv_shape);
    migraphx::shape data_shape{migraphx::shape::int64_type, {3, 4, 4}};
    auto input = mm->add_parameter("x", data_shape);
    mm->add_instruction(migraphx::make_op("fill"), dv, input);
    p.compile(migraphx::make_target("ref"));

    std::vector<int64_t> dv_data = {2};
    std::vector<int64_t> input_data(48);
    migraphx::parameter_map params;
    params["x"]  = migraphx::argument(data_shape, input_data.data());
    params["dv"] = migraphx::argument(dv_shape, dv_data.data());
    auto result  = p.eval(params).back();
    std::vector<int64_t> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<int64_t> gold(48, 2);
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
