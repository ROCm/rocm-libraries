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
#include <migraphx/pass_manager.hpp>
#include <migraphx/split_single_dyn_dim.hpp>
#include <migraphx/simplify_dyn_ops.hpp>
#include <migraphx/dead_code_elimination.hpp>

#include <test.hpp>

TEST_CASE(onehot0)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape inds_s{migraphx::shape::int64_type, {4}};
    migraphx::shape depth_s{migraphx::shape::int64_type, {1}};
    migraphx::shape values_s{migraphx::shape::float_type, {2}};
    auto inds_param   = mm->add_parameter("indices", inds_s);
    auto depth_param  = mm->add_parameter("depth", depth_s);
    auto values_param = mm->add_parameter("values", values_s);
    mm->add_instruction(
        migraphx::make_op("onehot", {{"axis", -1}}), inds_param, depth_param, values_param);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int64_t> indices_data = {0, -8, -1, 5};
    std::vector<int64_t> depth_data   = {3};
    std::vector<float> values_data    = {0.0, 5.0};
    params["indices"]                 = migraphx::argument(inds_s, indices_data.data());
    params["depth"]                   = migraphx::argument(depth_s, depth_data.data());
    params["values"]                  = migraphx::argument(values_s, values_data.data());
    auto result                       = p.eval(params).back();
    // clang-format off
    std::vector<float> gold =
    {
        5.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 5.0,
        0.0, 0.0, 0.0
    };
    // clang-format on
    std::vector<float> results_vector(4 * 3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(onehot1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape inds_s{migraphx::shape::int64_type, {2, 2}};
    migraphx::shape depth_s{migraphx::shape::int64_type, {1}};
    migraphx::shape values_s{migraphx::shape::float_type, {2}};
    auto inds_param   = mm->add_parameter("indices", inds_s);
    auto depth_lit    = mm->add_literal(migraphx::literal{depth_s, {3}});
    auto values_param = mm->add_parameter("values", values_s);
    mm->add_instruction(
        migraphx::make_op("onehot", {{"axis", -1}}), inds_param, depth_lit, values_param);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int64_t> indices_data = {0, 2, 1, -1};
    std::vector<float> values_data    = {0.0, 1.0};
    params["indices"]                 = migraphx::argument(inds_s, indices_data.data());
    params["values"]                  = migraphx::argument(values_s, values_data.data());
    auto result                       = p.eval(params).back();
    // clang-format off
    std::vector<float> gold =
    {
        1.0, 0.0, 0.0,
        0.0, 0.0, 1.0,

        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    // clang-format on
    std::vector<float> results_vector(2 * 2 * 3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(onehot_depth_attr)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape inds_s{migraphx::shape::int64_type, {2, 2}};
    migraphx::shape values_s{migraphx::shape::float_type, {2}};
    auto inds_param   = mm->add_parameter("indices", inds_s);
    auto values_param = mm->add_parameter("values", values_s);
    mm->add_instruction(
        migraphx::make_op("onehot", {{"axis", -1}, {"depth", 3}}), inds_param, values_param);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int64_t> indices_data = {0, 2, 1, -1};
    std::vector<float> values_data    = {0.0, 1.0};
    params["indices"]                 = migraphx::argument(inds_s, indices_data.data());
    params["values"]                  = migraphx::argument(values_s, values_data.data());
    auto result                       = p.eval(params).back();
    // clang-format off
    std::vector<float> gold =
    {
        1.0, 0.0, 0.0,
        0.0, 0.0, 1.0,

        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    // clang-format on
    std::vector<float> results_vector(2 * 2 * 3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(onehot_dyn)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape inds_s{migraphx::shape::int64_type, {{1, 4}, {2, 2}}};
    migraphx::shape depth_s{migraphx::shape::int64_type, {1}};
    migraphx::shape values_s{migraphx::shape::int32_type, {2}};
    auto inds_param   = mm->add_parameter("indices", inds_s);
    auto depth_lit    = mm->add_literal(migraphx::literal{depth_s, {3}});
    auto values_param = mm->add_parameter("values", values_s);
    mm->add_instruction(
        migraphx::make_op("onehot", {{"axis", -1}}), inds_param, depth_lit, values_param);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int64_t> indices_data = {0, 2, 1, -1};
    std::vector<int32_t> values_data  = {-3, 5};
    migraphx::shape static_inds_shape{migraphx::shape::int64_type, {2, 2}};
    params["indices"] = migraphx::argument(static_inds_shape, indices_data.data());
    params["values"]  = migraphx::argument(values_s, values_data.data());
    auto result       = p.eval(params).back();
    // clang-format off
    std::vector<int32_t> gold =
    {
        5, -3, -3,
        -3, -3, 5,
        -3, 5, -3,
        -3, -3, 5 
    };
    // clang-format on
    std::vector<int32_t> results_vector(2 * 2 * 3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(onehot_neg_depth_error)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape inds_s{migraphx::shape::int64_type, {4}};
    migraphx::shape depth_s{migraphx::shape::int64_type, {1}};
    migraphx::shape values_s{migraphx::shape::float_type, {2}};
    auto inds_param   = mm->add_parameter("indices", inds_s);
    auto depth_param  = mm->add_parameter("depth", depth_s);
    auto values_param = mm->add_parameter("values", values_s);
    mm->add_instruction(
        migraphx::make_op("onehot", {{"axis", -1}}), inds_param, depth_param, values_param);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int64_t> indices_data = {0, 2, -1, 5};
    std::vector<int64_t> depth_data   = {-2};
    std::vector<float> values_data    = {0.0, 5.0};
    params["indices"]                 = migraphx::argument(inds_s, indices_data.data());
    params["depth"]                   = migraphx::argument(depth_s, depth_data.data());
    params["values"]                  = migraphx::argument(values_s, values_data.data());
    EXPECT(test::throws([&] { std::ignore = p.eval(params).back(); }));
}

// make sure the result is the same when using the simplify pass
TEST_CASE(onehot_simplify_test0)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape inds_s{migraphx::shape::int64_type, {2, 2}};
    migraphx::shape depth_s{migraphx::shape::int64_type, {1}};
    migraphx::shape values_s{migraphx::shape::float_type, {2}};
    auto inds_param   = mm->add_parameter("indices", inds_s);
    auto depth_lit    = mm->add_literal(migraphx::literal{depth_s, {3}});
    auto values_param = mm->add_parameter("values", values_s);
    mm->add_instruction(
        migraphx::make_op("onehot", {{"axis", 0}}), inds_param, depth_lit, values_param);
    migraphx::run_passes(p, {migraphx::simplify_dyn_ops{}, migraphx::dead_code_elimination{}});
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int64_t> indices_data = {0, 2, 1, -1};
    std::vector<float> values_data    = {0.0, 1.0};
    params["indices"]                 = migraphx::argument(inds_s, indices_data.data());
    params["values"]                  = migraphx::argument(values_s, values_data.data());
    auto result                       = p.eval(params).back();
    // clang-format off
    std::vector<float> gold =
    {
        1.0, 0.0,
        0.0, 0.0,

        0.0, 0.0,
        1.0, 0.0,

        0.0, 1.0,
        0.0, 1.0
    };
    // clang-format on
    std::vector<float> results_vector(2 * 2 * 3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(onehot_simplify_test1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape inds_s{migraphx::shape::int64_type, {2, 2}};
    migraphx::shape depth_s{migraphx::shape::int64_type, {1}};
    migraphx::shape values_s{migraphx::shape::int32_type, {2}};
    auto inds_param   = mm->add_parameter("indices", inds_s);
    auto depth_lit    = mm->add_literal(migraphx::literal{depth_s, {3}});
    auto values_param = mm->add_parameter("values", values_s);
    mm->add_instruction(
        migraphx::make_op("onehot", {{"axis", -1}}), inds_param, depth_lit, values_param);
    migraphx::run_passes(p, {migraphx::simplify_dyn_ops{}, migraphx::dead_code_elimination{}});
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int64_t> indices_data = {0, 2, 1, -1};
    std::vector<int32_t> values_data  = {-3, 5};
    params["indices"]                 = migraphx::argument(inds_s, indices_data.data());
    params["values"]                  = migraphx::argument(values_s, values_data.data());
    auto result                       = p.eval(params).back();
    // clang-format off
    std::vector<int32_t> gold =
    {
        5, -3, -3,
        -3, -3, 5,
        -3, 5, -3,
        -3, -3, 5 
    };
    // clang-format on
    std::vector<int32_t> results_vector(2 * 2 * 3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(onehot_simplify_test2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape inds_s{migraphx::shape::int64_type, {4}};
    migraphx::shape depth_s{migraphx::shape::int64_type, {1}, {0}};
    migraphx::shape values_s{migraphx::shape::float_type, {2}};
    auto inds_param   = mm->add_parameter("indices", inds_s);
    auto depth_lit    = mm->add_literal(migraphx::literal{depth_s, {3}});
    auto values_param = mm->add_parameter("values", values_s);
    mm->add_instruction(
        migraphx::make_op("onehot", {{"axis", -1}}), inds_param, depth_lit, values_param);
    migraphx::run_passes(p, {migraphx::simplify_dyn_ops{}, migraphx::dead_code_elimination{}});
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int64_t> indices_data = {0, -8, -1, 5};
    std::vector<float> values_data    = {0.0, 5.0};
    params["indices"]                 = migraphx::argument(inds_s, indices_data.data());
    params["values"]                  = migraphx::argument(values_s, values_data.data());
    auto result                       = p.eval(params).back();
    // clang-format off
    std::vector<float> gold =
    {
        5.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 5.0,
        0.0, 0.0, 0.0
    };
    // clang-format on
    std::vector<float> results_vector(4 * 3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
