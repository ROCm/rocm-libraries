/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

static auto run_program(const migraphx::value& op, bool custom_idx = false, bool fill1 = false)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 5}};
    auto data                                          = mm->add_parameter("data", s);
    std::vector<migraphx::instruction_ref> topk_inputs = {data};
    if(custom_idx)
    {
        migraphx::shape is{migraphx::shape::uint16_type, {3, 5}};
        std::vector<int16_t> indices(is.elements());
        std::iota(indices.rbegin(), indices.rend(), 1);
        topk_inputs.push_back(mm->add_literal(migraphx::literal{is, indices}));
    }
    auto r  = mm->add_instruction(migraphx::make_op("topk", op), topk_inputs);
    auto r0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), r);
    auto r1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), r);
    mm->add_return({r0, r1});

    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data = {
        2.1,
        2.3,
        2.0,
        2.5,
        1.9,
        3.3,
        0.2,
        4.5,
        0.1,
        0.8,
        1.0,
        4.5,
        2.1,
        0.8,
        1.5,
    };
    if(fill1)
        input_data.assign(input_data.size(), 1);
    migraphx::parameter_map pp;
    pp["data"] = migraphx::argument(s, input_data.data());
    auto rets  = p.eval(pp);
    std::vector<float> ret_val;
    rets.front().visit([&](auto v) { ret_val.assign(v.begin(), v.end()); });
    std::vector<int64_t> ret_ind;
    rets.back().visit([&](auto v) { ret_ind.assign(v.begin(), v.end()); });

    return std::make_pair(ret_val, ret_ind);
}

TEST_CASE(topk_largest0)
{
    auto results                = run_program({{"axis", 0}, {"k", 2}, {"largest", 1}});
    std::vector<float> gold_val = {3.3, 4.5, 4.5, 2.5, 1.9, 2.1, 2.3, 2.1, 0.8, 1.5};
    EXPECT(results.first == gold_val);
    std::vector<int64_t> gold_ind = {1, 2, 1, 0, 0, 0, 0, 2, 2, 2};
    EXPECT(results.second == gold_ind);
}

TEST_CASE(topk_largest1)
{
    auto results                = run_program({{"axis", 1}, {"k", 4}, {"largest", 1}});
    std::vector<float> gold_val = {2.5, 2.3, 2.1, 2, 4.5, 3.3, 0.8, 0.2, 4.5, 2.1, 1.5, 1};
    EXPECT(results.first == gold_val);
    std::vector<int64_t> gold_ind = {3, 1, 0, 2, 2, 0, 4, 1, 1, 2, 4, 0};
    EXPECT(results.second == gold_ind);
}

TEST_CASE(topk_largest_same)
{
    auto results = run_program({{"axis", 1}, {"k", 4}, {"largest", 1}}, false, true);
    EXPECT(std::all_of(results.first.begin(), results.first.end(), [](auto i) {
        return migraphx::float_equal(i, 1);
    }));
    std::vector<int64_t> gold_ind = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    EXPECT(results.second == gold_ind);
}

TEST_CASE(topk_smallest1)
{
    auto results                = run_program({{"axis", 1}, {"k", 4}, {"largest", 0}});
    std::vector<float> gold_val = {1.9, 2, 2.1, 2.3, 0.1, 0.2, 0.8, 3.3, 0.8, 1, 1.5, 2.1};
    EXPECT(results.first == gold_val);
    std::vector<int64_t> gold_ind = {4, 2, 0, 1, 3, 1, 4, 0, 3, 0, 4, 2};
    EXPECT(results.second == gold_ind);
}

TEST_CASE(topk_smallest_same)
{
    auto results = run_program({{"axis", 1}, {"k", 4}, {"largest", 0}}, false, true);
    EXPECT(std::all_of(results.first.begin(), results.first.end(), [](auto i) {
        return migraphx::float_equal(i, 1);
    }));
    std::vector<int64_t> gold_ind = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    EXPECT(results.second == gold_ind);
}

TEST_CASE(topk_largest_custom_indices)
{
    auto results                = run_program({{"axis", 1}, {"k", 4}, {"largest", 1}}, true);
    std::vector<float> gold_val = {2.5, 2.3, 2.1, 2, 4.5, 3.3, 0.8, 0.2, 4.5, 2.1, 1.5, 1};
    EXPECT(results.first == gold_val);
    std::vector<int64_t> gold_ind = {12, 14, 15, 13, 8, 10, 6, 9, 4, 3, 1, 5};
    EXPECT(results.second == gold_ind);
}

TEST_CASE(topk_smallest_custom_indices)
{
    auto results                = run_program({{"axis", 1}, {"k", 4}, {"largest", 0}}, true);
    std::vector<float> gold_val = {1.9, 2, 2.1, 2.3, 0.1, 0.2, 0.8, 3.3, 0.8, 1, 1.5, 2.1};
    EXPECT(results.first == gold_val);
    std::vector<int64_t> gold_ind = {11, 13, 15, 14, 7, 9, 6, 10, 2, 5, 1, 3};
    EXPECT(results.second == gold_ind);
}
