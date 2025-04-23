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

#include <test.hpp>

TEST_CASE(bit_cast_fp8)
{
    using migraphx::fp8::fp8e4m3fn;
    using migraphx::fp8::fp8e4m3fnuz;
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::fp8e4m3fn_type, {2, 2}};
    std::vector<fp8e4m3fn> data;
    data.push_back(fp8e4m3fn{26.0f});
    data.push_back(fp8e4m3fn{3.0f});
    data.push_back(fp8e4m3fn{96.0f});
    data.push_back(fp8e4m3fn{-1.25f});
    auto lit = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(
        migraphx::make_op("bit_cast", {{"target_type", migraphx::shape::fp8e4m3fnuz_type}}), lit);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<fp8e4m3fnuz> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<fp8e4m3fnuz> gold;
    gold.push_back(fp8e4m3fnuz{13.0f});
    gold.push_back(fp8e4m3fnuz{1.5f});
    gold.push_back(fp8e4m3fnuz{48.0f});
    gold.push_back(fp8e4m3fnuz{-0.625f});
    EXPECT(results_vector == gold);
}

TEST_CASE(bit_cast_uint8)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int8_type, {2, 2}};
    std::vector<int8_t> data = {23, -3, 0, -1};
    auto lit                 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(
        migraphx::make_op("bit_cast", {{"target_type", migraphx::shape::uint8_type}}), lit);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<uint8_t> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<uint8_t> gold = {23, 253, 0, 255};
    EXPECT(results_vector == gold);
}
