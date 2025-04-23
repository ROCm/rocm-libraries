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

TEST_CASE(transpose_test)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 2, 2, 3}};
    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0);

    {
        migraphx::program p;
        auto* mm                  = p.get_main_module();
        auto l                    = mm->add_literal(migraphx::literal{a_shape, data});
        std::vector<int64_t> perm = {0, 3, 1, 2};
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), l);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
    }
    {
        migraphx::program p;
        auto* mm                  = p.get_main_module();
        auto l                    = mm->add_literal(migraphx::literal{a_shape, data});
        std::vector<int64_t> perm = {0, 3, 1, 2};
        auto result =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), l);
        mm->add_instruction(migraphx::make_op("contiguous"), result);
        p.compile(migraphx::make_target("ref"));
        auto result2 = p.eval({}).back();

        std::vector<float> results_vector(12);
        result2.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
        EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    }
}

TEST_CASE(transpose_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {2, 2}, {2, 2}, {3, 3}}};
    auto l                    = mm->add_parameter("X", s);
    std::vector<int64_t> perm = {0, 3, 1, 2};
    mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), l);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0);
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 2, 2, 3}};
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();

    std::vector<size_t> new_lens = {1, 3, 2, 2};
    EXPECT(result.get_shape().lens() == new_lens);

    std::vector<float> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
