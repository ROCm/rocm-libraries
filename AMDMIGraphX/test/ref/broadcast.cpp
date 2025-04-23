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

TEST_CASE(broadcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::int32_type, {2, 2}};
    std::vector<int32_t> a_data{0, 0, 0, 0};
    migraphx::shape b_shape{migraphx::shape::int32_type, {2}};
    std::vector<int32_t> b_data{-2, -3};
    uint64_t axis = 0;
    auto l1       = mm->add_literal(migraphx::literal{a_shape, a_data});
    auto l2       = mm->add_literal(migraphx::literal{b_shape, b_data});
    mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", l1->get_shape().lens()}}), l2);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    auto output = result.get<int32_t>();
    EXPECT(output(0, 0) == -2);
    EXPECT(output(0, 1) == -2);
    EXPECT(output(1, 0) == -3);
    EXPECT(output(1, 1) == -3);
}

TEST_CASE(broadcast_2in_static_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::int32_type, {2, 2}};
    std::vector<int32_t> a_data{0, 0, 0, 0};
    migraphx::shape b_shape{migraphx::shape::int32_type, {2}};
    std::vector<int32_t> b_data{-2, -3};
    uint64_t axis = 0;
    auto l1       = mm->add_literal(migraphx::literal{a_shape, a_data});
    auto l2       = mm->add_literal(migraphx::literal{b_shape, b_data});
    mm->add_instruction(migraphx::make_op("broadcast", {{"axis", axis}}), l2, l1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    auto output = result.get<int32_t>();
    EXPECT(output(0, 0) == -2);
    EXPECT(output(0, 1) == -2);
    EXPECT(output(1, 0) == -3);
    EXPECT(output(1, 1) == -3);
}

TEST_CASE(broadcast_2in_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::int32_type, {{2, 2}, {2, 4}}};
    migraphx::shape b_shape{migraphx::shape::int32_type, {2}};
    std::vector<int32_t> b_data{-2, -3};
    uint64_t axis = 0;
    auto pa       = mm->add_parameter("a", a_shape);
    auto lb       = mm->add_literal(migraphx::literal{b_shape, b_data});
    mm->add_instruction(migraphx::make_op("broadcast", {{"axis", axis}}), lb, pa);
    p.compile(migraphx::make_target("ref"));

    std::vector<int32_t> a_data{0, 0, 0, 0};
    migraphx::shape input_fixed_shape0{migraphx::shape::int32_type, {2, 2}};
    migraphx::parameter_map params0;
    params0["a"] = migraphx::argument(input_fixed_shape0, a_data.data());
    auto result  = p.eval(params0).back();
    auto output  = result.get<int32_t>();
    EXPECT(output(0, 0) == -2);
    EXPECT(output(0, 1) == -2);
    EXPECT(output(1, 0) == -3);
    EXPECT(output(1, 1) == -3);
}

TEST_CASE(isnan_broadcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::float_type, {3}};
    migraphx::shape s1{migraphx::shape::float_type, {3, 2}};
    auto nan_val             = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> data0 = {1.2, 5.2, nan_val};
    auto l0                  = mm->add_literal(migraphx::literal{s0, data0});
    auto l1                  = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", s1.lens()}}), l0);
    mm->add_instruction(migraphx::make_op("isnan"), l1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0, 0, 0, 0, 1, 1};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
