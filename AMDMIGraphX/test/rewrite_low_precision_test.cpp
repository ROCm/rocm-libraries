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
#include <migraphx/rewrite_low_precision.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/common.hpp>
#include <migraphx/verify.hpp>
#include <basic_ops.hpp>

#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::rewrite_low_precision{}, migraphx::dead_code_elimination{}});
}

template <migraphx::shape::type_t DType, typename T>
static void
create_pow2_div(migraphx::module& m, const std::vector<std::size_t>& input_lens, T divisor)
{
    migraphx::shape s_input{DType, input_lens};
    migraphx::shape s_lit{DType, {1}};
    auto l_pow_2 = m.add_literal(migraphx::literal{s_lit, {2.0f}});
    auto l_div   = m.add_literal(migraphx::literal{s_lit, {divisor}});

    auto input = m.add_parameter("input", s_input);
    auto pow   = add_common_op(m, migraphx::make_op("pow"), {input, l_pow_2});
    auto div   = add_common_op(m, migraphx::make_op("div"), {pow, l_div});
    m.add_return({div});
}

TEST_CASE(simplify_pow2_div)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::half_type, {1}});
        auto n   = m1.add_literal(migraphx::half{5.0f});
        auto two = m1.add_literal(migraphx::half{2.0f});

        auto pow = m1.add_instruction(migraphx::make_op("pow"), x, two);
        auto div = m1.add_instruction(migraphx::make_op("div"), pow, n);
        m1.add_instruction(pass_op{}, div);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::half_type, {1}});
        auto n = m2.add_literal(migraphx::half{5.0f});

        auto x_div_n = m2.add_instruction(migraphx::make_op("div"), x, n);
        auto mul     = m2.add_instruction(migraphx::make_op("mul"), x_div_n, x);
        m2.add_instruction(pass_op{}, mul);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(no_simplify_float_pow2_div)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::float_type, {1}});
        auto n   = m1.add_literal(5.0f);
        auto two = m1.add_literal(2.0f);

        auto pow = m1.add_instruction(migraphx::make_op("pow"), x, two);
        auto div = m1.add_instruction(migraphx::make_op("div"), pow, n);
        m1.add_instruction(pass_op{}, div);
    }
    auto m2 = m1;
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(no_simplify_int_pow2_div)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto n   = m1.add_literal(5);
        auto two = m1.add_literal(2);

        auto pow = m1.add_instruction(migraphx::make_op("pow"), x, two);
        auto div = m1.add_instruction(migraphx::make_op("div"), pow, n);
        m1.add_instruction(pass_op{}, div);
    }
    auto m2 = m1;
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(simplify_x_square_div)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::half_type, {1}});
        auto n = m1.add_literal(migraphx::half{5.0f});

        auto pow = m1.add_instruction(migraphx::make_op("mul"), x, x);
        auto div = m1.add_instruction(migraphx::make_op("div"), pow, n);
        m1.add_instruction(pass_op{}, div);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::half_type, {1}});
        auto n = m2.add_literal(migraphx::half{5.0f});

        auto x_div_n = m2.add_instruction(migraphx::make_op("div"), x, n);
        auto mul     = m2.add_instruction(migraphx::make_op("mul"), x_div_n, x);
        m2.add_instruction(pass_op{}, mul);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(no_simplify_x_mul_y_div)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::half_type, {1}});
        auto y = m1.add_parameter("y", {migraphx::shape::half_type, {1}});
        auto n = m1.add_literal(migraphx::half{5.0f});

        auto pow = m1.add_instruction(migraphx::make_op("mul"), x, y);
        auto div = m1.add_instruction(migraphx::make_op("div"), pow, n);
        m1.add_instruction(pass_op{}, div);
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(rewrite_pow2_div_fp16_accuracy_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s_input{migraphx::shape::half_type, {1, 3, 9}};
    create_pow2_div<migraphx::shape::half_type, migraphx::half>(
        *mm, s_input.lens(), migraphx::half{9.0});
    run_pass(*mm);
    p.compile(migraphx::make_target("ref"));

    // >>> np.random.seed(0)
    // >>> d = (np.sqrt(np.finfo(np.float16).max) * np.random.randn(27)).astype(np.float16)
    std::vector<float> tmp = {451.5,  102.4,  250.4, 573.5, 477.8, -250.,  243.1, -38.72, -26.4,
                              105.06, 36.84,  372.,  194.8, 31.14, 113.56, 85.4,  382.2,  -52.5,
                              80.1,   -218.5, -653., 167.2, 221.2, -189.9, 581.,  -372.2, 11.71};

    std::vector<migraphx::half> data = {tmp.begin(), tmp.end()};

    migraphx::parameter_map params;
    params["input"] = migraphx::argument(s_input, data.data());
    auto result     = p.eval(params).back();
    std::vector<migraphx::half> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    // The fp16 results without the rewrite should have `inf` values:
    // >>> (pow(d, 2) / 9.0)
    //     inf, 1164., 6964., inf, inf, 6944., 6568., 166.5, 77.5,
    //     1227., 150.8, inf, 4212., 107.75, 1433., 810., inf, 306.2,
    //     713.5, 5304., inf, 3108., 5440., 4008., inf, inf, 15.234

    // Expected results with rewrite:
    // >>> (pow(d.astype(np.float32), 2) / 9.0) .astype(np.float16)
    tmp = {22656., 1165., 6964.,  36544., 25360., 6944., 6568.,  166.6,  77.5,
           1226.,  150.9, 15376., 4216.,  107.75, 1433., 810.,   16232., 306.2,
           713.5,  5304., 47392., 3108.,  5440.,  4006., 37504., 15400., 15.24};

    std::vector<migraphx::half> gold = {tmp.begin(), tmp.end()};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(rewrite_pow2_div_fp16_same_result_test)
{
    migraphx::shape s_input{migraphx::shape::half_type, {1, 3, 9}};
    // >>> np.random.seed(42)
    // >>> d = (100 * np.random.randn(27)).astype(np.float16)
    std::vector<float> tmp{49.66, -13.83, 64.75,  152.2,  -23.42, -23.4,  157.9,  76.75,  -46.94,
                           54.25, -46.34, -46.56, 24.2,   -191.4, -172.5, -56.22, -101.3, 31.42,
                           -90.8, -141.2, 146.6,  -22.58, 6.754,  -142.5, -54.44, 11.09,  -115.1};
    std::vector<migraphx::half> data = {tmp.begin(), tmp.end()};

    migraphx::parameter_map params;
    params["input"] = migraphx::argument(s_input, data.data());

    migraphx::program p1;
    auto* mm = p1.get_main_module();
    create_pow2_div<migraphx::shape::half_type, migraphx::half>(
        *mm, s_input.lens(), migraphx::half{9.0f});
    run_pass(*mm);
    p1.compile(migraphx::make_target("ref"));
    auto result1 = p1.eval(params).back();
    std::vector<migraphx::half> rewrite_results;
    result1.visit([&](auto output) { rewrite_results.assign(output.begin(), output.end()); });

    migraphx::program p2;
    mm = p2.get_main_module();
    create_pow2_div<migraphx::shape::half_type, migraphx::half>(
        *mm, s_input.lens(), migraphx::half{9.0f});
    p2.compile(migraphx::make_target("ref"));
    auto result2 = p2.eval(params).back();
    std::vector<migraphx::half> no_rewrite_results;
    result2.visit([&](auto output) { no_rewrite_results.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(rewrite_results, no_rewrite_results));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
