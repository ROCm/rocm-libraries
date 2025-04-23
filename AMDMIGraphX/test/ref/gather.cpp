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

TEST_CASE(gather_non_std_test)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        std::vector<float> data = {0.5f, 3.5f, 6.5f, 1.5f, 4.5f, 7.5f, 2.5f, 2.5f, 8.5f};
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto d = mm->add_literal(migraphx::literal{s, data});
        migraphx::shape s_indices{migraphx::shape::int32_type, {2, 2}};
        std::vector<int> indices{-3, -3, -1, -1};
        auto ind = mm->add_literal(migraphx::literal{s_indices, indices});
        auto td = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), d);
        auto tind =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), ind);

        mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), td, tind);
        auto result               = p.eval({}).back();
        std::vector<float> golden = {
            0.5f, 1.5f, 2.5f, 6.5f, 7.5f, 8.5f, 0.5f, 1.5f, 2.5f, 6.5f, 7.5f, 8.5f};
        std::vector<float> res_data;
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify::verify_rms_range(res_data, golden));
    }
}

TEST_CASE(gather_test_1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    auto a0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape s_indices{migraphx::shape::int32_type, {1, 2}};
    std::vector<int> indices{0, 2};
    auto a1  = mm->add_literal(migraphx::literal{s_indices, indices});
    int axis = 0;
    mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> res_data(4 * 5);
    std::vector<float> golden = {0.5f, 1.5f, 2.5f, 6.5f, 7.5f, 8.5f};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(res_data, golden));
}

TEST_CASE(gather_test_2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    auto a0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape s_indices{migraphx::shape::int32_type, {1, 2}};
    std::vector<int> indices{-3, -1};
    auto a1  = mm->add_literal(migraphx::literal{s_indices, indices});
    int axis = 0;
    mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> res_data(4 * 5);
    std::vector<float> golden = {0.5f, 1.5f, 2.5f, 6.5f, 7.5f, 8.5f};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(res_data, golden));
}

TEST_CASE(gather_test_3)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    auto a0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape s_indices{migraphx::shape::int32_type, {1, 2}};
    std::vector<int> indices{0, 2};
    auto a1  = mm->add_literal(migraphx::literal{s_indices, indices});
    int axis = 1;
    mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> res_data(4 * 5);
    std::vector<float> golden = {0.5f, 2.5f, 3.5f, 5.5f, 6.5f, 8.5f};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(res_data, golden));
}

TEST_CASE(gather_test_4)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    auto a0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape s_indices{migraphx::shape::int32_type, {1, 2}};
    std::vector<int> indices{0, 2};
    auto a1  = mm->add_literal(migraphx::literal{s_indices, indices});
    int axis = -1;
    mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> res_data(4 * 5);
    std::vector<float> golden = {0.5f, 2.5f, 3.5f, 5.5f, 6.5f, 8.5f};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(res_data, golden));
}

TEST_CASE(gather_test_5)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    auto a0 = mm->add_literal(migraphx::literal{s, data});
    // scalar index
    migraphx::shape s_indices{migraphx::shape::int32_type};
    std::vector<int> indices{0};
    auto a1  = mm->add_literal(migraphx::literal{s_indices, indices});
    int axis = -1;
    mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> res_data{};
    std::vector<float> golden = {0.5f, 3.5f, 6.5f};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(res_data, golden));
}

TEST_CASE(gather_test_6)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    auto a0 = mm->add_literal(migraphx::literal{s, data});
    // scalar index
    migraphx::shape s_indices{migraphx::shape::int32_type};
    std::vector<int> indices{-3};
    auto a1  = mm->add_literal(migraphx::literal{s_indices, indices});
    int axis = -1;
    mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> res_data{};
    std::vector<float> golden = {0.5f, 3.5f, 6.5f};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(res_data, golden));
}

TEST_CASE(gather_test_7)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto a0 = mm->add_literal(migraphx::literal{s, data});
    // scalar index
    migraphx::shape s_indices{migraphx::shape::int32_type};
    std::vector<int> indices{0};
    auto a1  = mm->add_literal(migraphx::literal{s_indices, indices});
    int axis = -1;
    mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> res_data{};
    std::vector<float> golden = {0.5f};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(res_data, golden));
}

TEST_CASE(gather_dyn_test0)
{
    // Dynamic data, static indices
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {{2, 5}, {3, 3}}};

    auto x = mm->add_parameter("x", s);
    std::vector<int> indices{1, 2};

    migraphx::shape s_ind{migraphx::shape::int32_type, {1, 2}};
    auto ind = mm->add_parameter("indices", s_ind);
    mm->add_instruction(migraphx::make_op("gather", {{"axis", 1}}), x, ind);

    migraphx::shape sresult{migraphx::shape::int32_type, {{2, 5}, {1, 1}, {2, 2}}};
    EXPECT(p.get_output_shapes().back() == sresult);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_fixed_shape{migraphx::shape::int32_type, {2, 3}};
    migraphx::shape input_indices{migraphx::shape::int32_type, {1, 2}};
    migraphx::parameter_map params;
    std::vector<int> data(2 * 3);
    std::iota(data.begin(), data.end(), 0);
    params["x"]       = migraphx::argument(input_fixed_shape, data.data());
    params["indices"] = migraphx::argument(input_indices, indices.data());
    auto result       = p.eval(params).back();

    std::vector<int> gold = {1, 2, 4, 5};
    std::vector<int> results_vector(2 * 1 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    migraphx::shape sfinal{migraphx::shape::int32_type, {2, 1, 2}};
    EXPECT(result.get_shape() == sfinal);
}

TEST_CASE(gather_dyn_test1)
{
    // Dynamic data, dynamic indices
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {{2, 5}, {4, 4}}};

    auto x = mm->add_parameter("x", s);

    migraphx::shape s_ind{migraphx::shape::int32_type, {{1, 8, {7}}, {2, 3, {3}}}};
    auto ind = mm->add_parameter("indices", s_ind);
    mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), x, ind);

    migraphx::shape sresult{migraphx::shape::int32_type, {{1, 8, {7}}, {2, 3, {3}}, {4, 4}}};
    EXPECT(p.get_output_shapes().back() == sresult);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_fixed_shape{migraphx::shape::int32_type, {3, 4}};
    migraphx::shape input_indices_shape{migraphx::shape::int32_type, {1, 2}};
    std::vector<int> indices{2, 0};
    migraphx::parameter_map params;

    std::vector<int> data(3 * 4);
    std::iota(data.begin(), data.end(), 0);
    params["x"]       = migraphx::argument(input_fixed_shape, data.data());
    params["indices"] = migraphx::argument(input_indices_shape, indices.data());
    auto result       = p.eval(params).back();

    std::vector<int> gold = {8, 9, 10, 11, 0, 1, 2, 3};
    std::vector<int> results_vector(1 * 2 * 4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    migraphx::shape sfinal{migraphx::shape::int32_type, {1, 2, 4}};
    EXPECT(result.get_shape() == sfinal);
}
