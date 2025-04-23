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

TEST_CASE(concat_test_1)
{
    migraphx::program p;
    auto* mm               = p.get_main_module();
    int axis               = 1;
    std::vector<int> data0 = {0, 1, 5, 6};
    std::vector<int> data1 = {2, 3, 4, 7, 8, 9};
    std::vector<int> data2 = {10, 20};
    migraphx::shape s0{migraphx::shape::int32_type, {2, 2}};
    migraphx::shape s1{migraphx::shape::int32_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::int32_type, {2, 1}};
    auto l0 = mm->add_literal(migraphx::literal{s0, data0});
    auto l1 = mm->add_literal(migraphx::literal{s1, data1});
    auto l2 = mm->add_literal(migraphx::literal{s2, data2});
    mm->add_instruction(migraphx::make_op("concat", {{"axis", axis}}), l0, l1, l2);
    p.compile(migraphx::make_target("ref"));
    auto result           = p.eval({}).back();
    std::vector<int> gold = {0, 1, 2, 3, 4, 10, 5, 6, 7, 8, 9, 20};
    std::vector<int> results_vector(2 * 6);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    EXPECT(migraphx::verify::verify_rms_range(result.get_shape().lens(),
                                              std::vector<std::size_t>({2, 6})));
    EXPECT(migraphx::verify::verify_rms_range(result.get_shape().strides(),
                                              std::vector<std::size_t>({6, 1})));
}

TEST_CASE(concat_test_2)
{
    migraphx::program p;
    auto* mm               = p.get_main_module();
    int axis               = -1;
    std::vector<int> data0 = {0, 1, 5, 6};
    std::vector<int> data1 = {2, 3, 4, 7, 8, 9};
    std::vector<int> data2 = {10, 20};
    migraphx::shape s0{migraphx::shape::int32_type, {2, 2}};
    migraphx::shape s1{migraphx::shape::int32_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::int32_type, {2, 1}};
    auto l0 = mm->add_literal(migraphx::literal{s0, data0});
    auto l1 = mm->add_literal(migraphx::literal{s1, data1});
    auto l2 = mm->add_literal(migraphx::literal{s2, data2});
    mm->add_instruction(migraphx::make_op("concat", {{"axis", axis}}), l0, l1, l2);
    p.compile(migraphx::make_target("ref"));
    auto result           = p.eval({}).back();
    std::vector<int> gold = {0, 1, 2, 3, 4, 10, 5, 6, 7, 8, 9, 20};
    std::vector<int> results_vector(2 * 6);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    EXPECT(migraphx::verify::verify_rms_range(result.get_shape().lens(),
                                              std::vector<std::size_t>({2, 6})));
    EXPECT(migraphx::verify::verify_rms_range(result.get_shape().strides(),
                                              std::vector<std::size_t>({6, 1})));
}

TEST_CASE(concat_test_3)
{
    migraphx::program p;
    auto* mm               = p.get_main_module();
    int axis               = 0;
    std::vector<int> data0 = {0, 1, 2, 3};
    std::vector<int> data1 = {4, 5, 6, 7, 8, 9};
    std::vector<int> data2 = {10, 11};
    migraphx::shape s0{migraphx::shape::int32_type, {2, 2}};
    migraphx::shape s1{migraphx::shape::int32_type, {3, 2}};
    migraphx::shape s2{migraphx::shape::int32_type, {1, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s0, data0});
    auto l1 = mm->add_literal(migraphx::literal{s1, data1});
    auto l2 = mm->add_literal(migraphx::literal{s2, data2});
    mm->add_instruction(migraphx::make_op("concat", {{"axis", axis}}), l0, l1, l2);
    p.compile(migraphx::make_target("ref"));
    auto result           = p.eval({}).back();
    std::vector<int> gold = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<int> results_vector(6 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    EXPECT(migraphx::verify::verify_rms_range(result.get_shape().lens(),
                                              std::vector<std::size_t>({6, 2})));
    EXPECT(migraphx::verify::verify_rms_range(result.get_shape().strides(),
                                              std::vector<std::size_t>({2, 1})));
}

TEST_CASE(concat_test_4)
{
    migraphx::program p;
    auto* mm               = p.get_main_module();
    int axis               = -2;
    std::vector<int> data0 = {0, 1, 2, 3};
    std::vector<int> data1 = {4, 5, 6, 7, 8, 9};
    std::vector<int> data2 = {10, 11};
    migraphx::shape s0{migraphx::shape::int32_type, {2, 2}};
    migraphx::shape s1{migraphx::shape::int32_type, {3, 2}};
    migraphx::shape s2{migraphx::shape::int32_type, {1, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s0, data0});
    auto l1 = mm->add_literal(migraphx::literal{s1, data1});
    auto l2 = mm->add_literal(migraphx::literal{s2, data2});
    mm->add_instruction(migraphx::make_op("concat", {{"axis", axis}}), l0, l1, l2);
    p.compile(migraphx::make_target("ref"));
    auto result           = p.eval({}).back();
    std::vector<int> gold = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<int> results_vector(6 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    EXPECT(migraphx::verify::verify_rms_range(result.get_shape().lens(),
                                              std::vector<std::size_t>({6, 2})));
    EXPECT(migraphx::verify::verify_rms_range(result.get_shape().strides(),
                                              std::vector<std::size_t>({2, 1})));
}

TEST_CASE(concat_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    int axis = 0;
    migraphx::shape s0{migraphx::shape::int32_type, {{2, 4, {2}}, {2, 3, {2}}}};
    migraphx::shape s1{migraphx::shape::int32_type, {{3, 4, {4}}, {2, 3, {2}}}};
    migraphx::shape s2{migraphx::shape::int32_type, {{1, 5, {3}}, {2, 3, {2}}}};

    auto input0 = mm->add_parameter("X", s0);
    auto input1 = mm->add_parameter("Y", s1);
    auto input2 = mm->add_parameter("Z", s2);
    mm->add_instruction(migraphx::make_op("concat", {{"axis", axis}}), input0, input1, input2);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape static_shape0{migraphx::shape::int32_type, {2, 2}};
    migraphx::shape static_shape1{migraphx::shape::int32_type, {3, 2}};
    migraphx::shape static_shape2{migraphx::shape::int32_type, {1, 2}};
    std::vector<int> data0 = {0, 1, 2, 3};
    std::vector<int> data1 = {4, 5, 6, 7, 8, 9};
    std::vector<int> data2 = {10, 11};
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(static_shape0, data0.data());
    params["Y"] = migraphx::argument(static_shape1, data1.data());
    params["Z"] = migraphx::argument(static_shape2, data2.data());
    auto result = p.eval(params).back();

    std::vector<int> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    EXPECT(migraphx::verify::verify_rms_range(result.get_shape().lens(),
                                              std::vector<std::size_t>({6, 2})));
}
