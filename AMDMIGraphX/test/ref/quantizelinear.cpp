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
#include <algorithm>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(quantizelinear_1)
{
    migraphx::shape xs{migraphx::shape::float_type, {2, 3, 3}};
    std::vector<float> xv = {
        -300, 600, 129, -1000, 4, 3, -6, 600, 550, -300, 600, 129, -1000, 4, 3, -6, 600, 550};
    migraphx::shape ss{migraphx::shape::float_type, {2, 3, 3}};
    std::vector<float> sv = {2, 2, 2, 4, 4, 4, 6, 6, 6, 2, 2, 2, 4, 4, 4, 6, 6, 6};
    migraphx::shape zs{migraphx::shape::int8_type, {2, 3, 3}};
    std::vector<uint8_t> zv = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    auto create_program     = [&]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_literal(xs, xv);
        auto s   = mm->add_literal(ss, sv);
        auto z   = mm->add_literal(zs, zv);
        mm->add_instruction(migraphx::make_op("quantizelinear"), x, s, z);
        return p;
    };

    migraphx::program p1 = create_program();
    p1.compile(migraphx::make_target("ref"));
    auto result = p1.eval({}).back();
    std::vector<float> results_vector(18);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{
        -128, 127, 64, -128, 1, 1, -1, 100, 92, -128, 127, 64, -128, 1, 1, -1, 100, 92};
    EXPECT(results_vector == gold);
}

TEST_CASE(quantizelinear_2)
{
    migraphx::shape xs{migraphx::shape::float_type, {2, 3, 3}};
    std::vector<float> xv = {
        -300, 600, 129, -1000, 4, 3, -6, 600, 550, -300, 600, 129, -1000, 4, 3, -6, 600, 550};
    migraphx::shape ss{migraphx::shape::float_type, {2, 3, 3}};
    std::vector<float> sv = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    auto create_program   = [&]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_literal(xs, xv);
        auto s   = mm->add_literal(ss, sv);
        mm->add_instruction(migraphx::make_op("quantizelinear"), x, s);
        return p;
    };

    migraphx::program p1 = create_program();
    p1.compile(migraphx::make_target("ref"));
    auto result = p1.eval({}).back();
    std::vector<float> results_vector(18);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0, 255, 64, 0, 2, 2, 0, 255, 255, 0, 255, 64, 0, 2, 2, 0, 255, 255};
    EXPECT(results_vector == gold);
}

template <class DType>
static void quantizelinear_fp8e4m3()
{
    migraphx::shape xs{migraphx::shape::float_type, {2, 2, 2}};
    migraphx::shape zs{migraphx::shape::get_type<DType>{}, {2, 2, 2}};
    std::vector<float> xv = {0.5, 0.75, -0.4375, 0.6875, -0.9375, -0.9375, 0.625, -0.5625};
    std::vector<float> sv = {0.25, 0.75, 0.5625, 0.4375, 0.8125, -0.6875, 0.875, -0.0625};
    std::vector<float> tmp = {0.6875, 0.75, -0.75, 0.5, -0.0625, 0.0625, -0.375, 0.25};
    std::vector<DType> zero_pts;
    std::transform(
        tmp.begin(), tmp.end(), std::back_inserter(zero_pts), [](auto x) { return DType(x); });
    auto create_program = [&]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_literal(xs, xv);
        auto s   = mm->add_literal(xs, sv);
        auto z   = mm->add_literal(zs, zero_pts);
        mm->add_instruction(migraphx::make_op("quantizelinear"), x, s, z);
        return p;
    };

    migraphx::program p1 = create_program();
    p1.compile(migraphx::make_target("ref"));
    auto result = p1.eval({}).back();
    std::vector<DType> results_vector(8);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<DType> gold;
    auto min_value = std::numeric_limits<DType>::lowest();
    auto max_value = std::numeric_limits<DType>::max();
    for(int i = 0; i < xv.size(); ++i)
    {
        double quantized = xv.at(i) / sv.at(i) + zero_pts.at(i);
        quantized        = std::max(static_cast<double>(min_value),
                             std::min(static_cast<double>(max_value), quantized));
        gold.push_back(DType(quantized));
    }
    EXPECT(results_vector == gold);
}
TEST_CASE_REGISTER(quantizelinear_fp8e4m3<migraphx::fp8::fp8e4m3fnuz>);
TEST_CASE_REGISTER(quantizelinear_fp8e4m3<migraphx::fp8::fp8e4m3fn>);

template <class DType>
static void quantizelinear_fp8e5m2()
{
    migraphx::shape xs{migraphx::shape::float_type, {2, 2, 2}};
    migraphx::shape zs{migraphx::shape::get_type<DType>{}, {2, 2, 2}};
    std::vector<float> xv = {0.5, 0.75, -0.4375, 0.625, -0.875, -0.875, 0.625, -0.5};
    std::vector<float> sv = {0.25, 0.75, 0.5, 0.4375, 0.875, -0.625, 0.875, -0.0625};
    std::vector<float> tmp = {0.6875, 0.75, -0.75, 0.5, -0.0625, 0.0625, -0.375, 0.25};
    std::vector<DType> zero_pts;
    std::transform(
        tmp.begin(), tmp.end(), std::back_inserter(zero_pts), [](auto x) { return DType(x); });
    auto create_program = [&]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_literal(xs, xv);
        auto s   = mm->add_literal(xs, sv);
        auto z   = mm->add_literal(zs, zero_pts);
        mm->add_instruction(migraphx::make_op("quantizelinear"), x, s, z);
        return p;
    };

    migraphx::program p1 = create_program();
    p1.compile(migraphx::make_target("ref"));
    auto result = p1.eval({}).back();
    std::vector<DType> results_vector(8);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<DType> gold;
    auto min_value = std::numeric_limits<DType>::lowest();
    auto max_value = std::numeric_limits<DType>::max();
    for(int i = 0; i < xv.size(); ++i)
    {
        double quantized = xv.at(i) / sv.at(i);
        quantized        = std::max(static_cast<double>(min_value),
                             std::min(static_cast<double>(max_value), quantized));
        gold.push_back(DType(quantized + zero_pts.at(i)));
    }
    EXPECT(results_vector == gold);
}
TEST_CASE_REGISTER(quantizelinear_fp8e5m2<migraphx::fp8::fp8e5m2fnuz>);
TEST_CASE_REGISTER(quantizelinear_fp8e5m2<migraphx::fp8::fp8e5m2>);
