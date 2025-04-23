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

TEST_CASE(quant_conv2d_padding_stride_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::int8_type, {2, 3, 4, 4}};
    std::vector<int8_t> a(2 * 3 * 4 * 4);
    std::iota(a.begin(), a.end(), 0);
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    migraphx::shape c_shape{migraphx::shape::int8_type, {2, 3, 3, 3}};
    std::vector<int8_t> c(2 * 3 * 3 * 3);
    std::iota(c.begin(), c.end(), 0);
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});
    mm->add_instruction(
        migraphx::make_op("quant_convolution", {{"padding", {1, 1}}, {"stride", {2, 2}}}), al, cl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<int32_t> gold = {4521,
                                 7014,
                                 7830,
                                 11952,
                                 10515,
                                 16734,
                                 19737,
                                 30906,
                                 13161,
                                 19542,
                                 19494,
                                 28800,
                                 34707,
                                 52590,
                                 54729,
                                 82746};
    std::vector<int32_t> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(quant_conv2d_padding_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::int8_type, {2, 3, 4, 4}};
    std::vector<int8_t> a(2 * 3 * 4 * 4);
    std::iota(a.begin(), a.end(), 0);
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    migraphx::shape c_shape{migraphx::shape::int8_type, {2, 3, 3, 3}};
    std::vector<int8_t> c(2 * 3 * 3 * 3);
    std::iota(c.begin(), c.end(), 0);
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});
    mm->add_instruction(
        migraphx::make_op("quant_convolution", {{"padding", {1, 1}}, {"stride", {1, 1}}}), al, cl);
    p.compile(migraphx::make_target("ref"));
    auto result               = p.eval({}).back();
    std::vector<int32_t> gold = {
        4521,  6753,  7014,  4635,  6858,  10197, 10548, 6939,  7830,  11601, 11952, 7839,  5007,
        7383,  7590,  4953,  10515, 15987, 16734, 11277, 16821, 25506, 26586, 17874, 19737, 29826,
        30906, 20718, 13593, 20505, 21198, 14187, 13161, 19281, 19542, 12699, 18522, 27045, 27396,
        17739, 19494, 28449, 28800, 18639, 11919, 17319, 17526, 11289, 34707, 51843, 52590, 34893,
        51813, 77346, 78426, 52002, 54729, 81666, 82746, 54846, 36057, 53769, 54462, 36075};

    std::vector<int32_t> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(quant_conv2d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::int8_type, {2, 3, 4, 4}};
    std::vector<int8_t> a(2 * 3 * 4 * 4);
    std::iota(a.begin(), a.end(), 0);
    auto al = mm->add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::int8_type, {2, 3, 3, 3}};
    std::vector<int8_t> c(2 * 3 * 3 * 3);
    std::iota(c.begin(), c.end(), 0);
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});

    mm->add_instruction(migraphx::make_op("quant_convolution"), al, cl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<int32_t> gold = {10197,
                                 10548,
                                 11601,
                                 11952,
                                 25506,
                                 26586,
                                 29826,
                                 30906,
                                 27045,
                                 27396,
                                 28449,
                                 28800,
                                 77346,
                                 78426,
                                 81666,
                                 82746};

    std::vector<int32_t> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
