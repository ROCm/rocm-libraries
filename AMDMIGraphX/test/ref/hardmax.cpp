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
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(hardmax_test_0)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<float> res_gold = {1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0,
                                   0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1};
    std::vector<std::size_t> input_lens{2, 3, 4};
    auto input_type = migraphx::shape::float_type;
    migraphx::shape data_shape{input_type, input_lens};
    auto input = mm->add_literal(migraphx::literal{data_shape, data});

    auto indices = mm->add_instruction(migraphx::make_op("argmax", {{"axis", 0}}), input);
    auto zero_data =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                            mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {0}}));
    auto updates = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", indices->get_shape().lens()}}),
        mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {1}}));
    mm->add_instruction(
        migraphx::make_op("scatter_none", {{"axis", 0}}), zero_data, indices, updates);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(result_vec, res_gold));
}

TEST_CASE(hardmax_test_1)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<float> res_gold = {1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                                   0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1};
    std::vector<std::size_t> input_lens{2, 3, 4};
    auto input_type = migraphx::shape::float_type;
    migraphx::shape data_shape{input_type, input_lens};
    auto input = mm->add_literal(migraphx::literal{data_shape, data});

    auto indices = mm->add_instruction(migraphx::make_op("argmax", {{"axis", 1}}), input);
    auto zero_data =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                            mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {0}}));
    auto updates = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", indices->get_shape().lens()}}),
        mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {1}}));
    mm->add_instruction(
        migraphx::make_op("scatter_none", {{"axis", 1}}), zero_data, indices, updates);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(result_vec, res_gold));
}

TEST_CASE(hardmax_test_2)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<float> res_gold = {0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                                   0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    std::vector<std::size_t> input_lens{2, 3, 4};
    auto input_type = migraphx::shape::float_type;
    migraphx::shape data_shape{input_type, input_lens};
    auto input = mm->add_literal(migraphx::literal{data_shape, data});

    auto indices = mm->add_instruction(migraphx::make_op("argmax", {{"axis", 2}}), input);
    auto zero_data =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                            mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {0}}));
    auto updates = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", indices->get_shape().lens()}}),
        mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {1}}));
    mm->add_instruction(
        migraphx::make_op("scatter_none", {{"axis", 2}}), zero_data, indices, updates);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(result_vec, res_gold));
}

TEST_CASE(hardmax_test_neg_2)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<float> res_gold = {1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                                   0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1};
    std::vector<std::size_t> input_lens{2, 3, 4};
    auto input_type = migraphx::shape::float_type;
    migraphx::shape data_shape{input_type, input_lens};
    auto input = mm->add_literal(migraphx::literal{data_shape, data});

    auto indices = mm->add_instruction(migraphx::make_op("argmax", {{"axis", -2}}), input);
    auto zero_data =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                            mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {0}}));
    auto updates = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", indices->get_shape().lens()}}),
        mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {1}}));
    mm->add_instruction(
        migraphx::make_op("scatter_none", {{"axis", -2}}), zero_data, indices, updates);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(result_vec, res_gold));
}

TEST_CASE(hardmax_test_neg_1)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<float> res_gold = {0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                                   0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    std::vector<std::size_t> input_lens{2, 3, 4};
    auto input_type = migraphx::shape::float_type;
    migraphx::shape data_shape{input_type, input_lens};
    auto input = mm->add_literal(migraphx::literal{data_shape, data});

    auto indices = mm->add_instruction(migraphx::make_op("argmax", {{"axis", -1}}), input);
    auto zero_data =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                            mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {0}}));
    auto updates = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", indices->get_shape().lens()}}),
        mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {1}}));
    mm->add_instruction(
        migraphx::make_op("scatter_none", {{"axis", -1}}), zero_data, indices, updates);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(result_vec, res_gold));
}
