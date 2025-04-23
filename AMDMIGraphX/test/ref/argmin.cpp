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
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(argmin_test_0)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<int64_t> res_gold = {1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    mm->add_instruction(migraphx::make_op("argmin", {{"axis", 0}}), dl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(result_vec, res_gold));
}

TEST_CASE(argmin_test_1)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<int64_t> res_gold = {2, 2, 0, 2, 0, 1, 2, 0};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    mm->add_instruction(migraphx::make_op("argmin", {{"axis", 1}}), dl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(result_vec, res_gold));
}

TEST_CASE(argmin_test_2)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<int64_t> res_gold = {2, 1, 0, 3, 3, 2};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    mm->add_instruction(migraphx::make_op("argmin", {{"axis", 2}}), dl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(result_vec, res_gold));
}

TEST_CASE(argmin_test_neg_1)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<int64_t> res_gold = {2, 1, 0, 3, 3, 2};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    mm->add_instruction(migraphx::make_op("argmin", {{"axis", -1}}), dl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(result_vec, res_gold));
}

TEST_CASE(argmin_test_nonstd_shape)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto dl = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 3, 4}}));
    auto dl_trans =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 2, 0}}}), dl);
    mm->add_instruction(migraphx::make_op("argmin", {{"axis", -1}}), dl_trans);
    auto p_uncompiled = p;
    p.compile(migraphx::make_target("ref"));
    auto result   = p.eval({}).back();
    auto res_gold = p_uncompiled.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });
    std::vector<int64_t> res_gold_vec;
    res_gold.visit([&](auto output) { res_gold_vec.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(result_vec, res_gold_vec));
}

TEST_CASE(argmin_test_select_last_index_0)
{
    migraphx::program p;
    auto* mm                      = p.get_main_module();
    std::vector<float> data       = {-2.0305, 0.853, -2.0305, 1.5706, 0.7545, 0.7545};
    std::vector<int64_t> res_gold = {2, 2};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    mm->add_instruction(migraphx::make_op("argmin", {{"axis", 1}, {"select_last_index", true}}),
                        dl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(result_vec, res_gold));
}

TEST_CASE(argmin_test_select_last_index_1)
{
    migraphx::program p;
    auto* mm                      = p.get_main_module();
    std::vector<float> data       = {-2.0305, 0.853, -2.0305, 1.5706, 0.7545, 0.7545};
    std::vector<int64_t> res_gold = {0, 1};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    mm->add_instruction(migraphx::make_op("argmin", {{"axis", 1}, {"select_last_index", false}}),
                        dl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(result_vec, res_gold));
}
