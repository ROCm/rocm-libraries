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

TEST_CASE(where_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sb{migraphx::shape::bool_type, {3, 3}};
    migraphx::shape sx{migraphx::shape::float_type, {3, 3}};

    std::vector<bool> b{true, true, true, false, false, false, true, false, true};
    std::vector<float> x(9, 1.0);
    std::vector<float> y(9, 2.0);

    auto lb = mm->add_literal(migraphx::literal{sb, b});
    auto lx = mm->add_literal(migraphx::literal{sx, x});
    auto ly = mm->add_literal(migraphx::literal{sx, y});
    auto w  = mm->add_instruction(migraphx::make_op("where"), lb, lx, ly);
    mm->add_return({w});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });
    std::vector<float> gold(9);
    for(int i = 0; i < gold.size(); ++i)
        gold[i] = b[i] ? x[i] : y[i];

    EXPECT(migraphx::verify::verify_rms_range(result_vec, gold));
}

TEST_CASE(where_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sb{migraphx::shape::bool_type, {{2, 3}, {2, 3}}};
    migraphx::shape sx{migraphx::shape::float_type, {{2, 3}, {2, 3}}};

    auto lb = mm->add_parameter("predicate", sb);
    auto lx = mm->add_parameter("X", sx);
    auto ly = mm->add_parameter("Y", sx);
    mm->add_instruction(migraphx::make_op("where"), lb, lx, ly);
    p.compile(migraphx::make_target("ref"));

    std::vector<char> b{1, 1, 1, 0, 0, 0, 1, 0, 1};
    std::vector<float> x(9, 1.0);
    std::vector<float> y(9, 2.0);
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3, 3}};
    migraphx::shape input_fixed_shape1{migraphx::shape::uint8_type, {3, 3}};
    params["X"] = migraphx::argument(input_fixed_shape0, x.data());
    params["Y"] = migraphx::argument(input_fixed_shape0, y.data());

    params["predicate"] = migraphx::argument(input_fixed_shape1, b.data());

    auto result = p.eval(params).back();
    std::vector<float> results_vector(3 * 3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1, 1, 1, 2, 2, 2, 1, 2, 1};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(where_broadcasted_inputs_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sb{migraphx::shape::bool_type, {3, 3}};

    std::vector<bool> b{true, true, true, false, false, false, true, false, true};

    auto lb  = mm->add_literal(migraphx::literal{sb, b});
    auto lx  = mm->add_literal(migraphx::literal(1.0f));
    auto ly  = mm->add_literal(migraphx::literal(2.0f));
    auto mbx = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 3}}}), lx);
    auto mby = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 3}}}), ly);
    auto w   = mm->add_instruction(migraphx::make_op("where"), lb, mbx, mby);
    mm->add_return({w});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });
    std::vector<float> gold(9);
    std::vector<float> x(9, 1.0);
    std::vector<float> y(9, 2.0);
    for(int i = 0; i < gold.size(); ++i)
        gold[i] = b[i] ? x[i] : y[i];

    EXPECT(migraphx::verify::verify_rms_range(result_vec, gold));
}
