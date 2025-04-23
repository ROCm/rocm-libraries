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

TEST_CASE(dequantizelinear_unsigned_int8)

{ /*uint8*/
    migraphx::shape xs{migraphx::shape::uint8_type, {1, 3, 3}};
    std::vector<uint8_t> xv = {0, 1, 2, 5, 10, 50, 100, 150, 250};
    migraphx::shape ss{migraphx::shape::float_type, {1, 3, 3}};
    std::vector<float> sv = {2, 2, 2, 2, 2, 2, 2, 2, 2};
    migraphx::shape zs{migraphx::shape::uint8_type, {1, 3, 3}};
    std::vector<uint8_t> zv = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    auto create_program     = [&]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_literal(xs, xv);
        auto s   = mm->add_literal(ss, sv);
        auto z   = mm->add_literal(zs, zv);
        mm->add_instruction(migraphx::make_op("dequantizelinear"), x, s, z);
        return p;
    };

    migraphx::program p1 = create_program();
    p1.compile(migraphx::make_target("ref"));
    auto result = p1.eval({}).back();
    std::vector<float> results_vector(9);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0, 2, 4, 10, 20, 100, 200, 300, 500};
    EXPECT(results_vector == gold);
}

TEST_CASE(dequantizelinear_signed_int8)
{ /*int8*/
    migraphx::shape xs{migraphx::shape::int8_type, {1, 3, 3}};
    std::vector<int8_t> xv = {-128, -100, -50, -1, 0, 1, 50, 100, 127};
    migraphx::shape ss{migraphx::shape::float_type, {1, 3, 3}};
    std::vector<float> sv = {2, 2, 2, 2, 2, 2, 2, 2, 2};
    auto create_program   = [&]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_literal(xs, xv);
        auto s   = mm->add_literal(ss, sv);
        mm->add_instruction(migraphx::make_op("dequantizelinear"), x, s);
        return p;
    };

    migraphx::program p1 = create_program();
    p1.compile(migraphx::make_target("ref"));
    auto result = p1.eval({}).back();
    std::vector<float> results_vector(9);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{-256, -200, -100, -2, 0, 2, 100, 200, 254};
    EXPECT(results_vector == gold);
}
