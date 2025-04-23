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
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <random>

#include <test.hpp>

/**
 * Reference test for the random_seed operation
 */
TEST_CASE(random_seed_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_instruction(migraphx::make_op("random_seed"));

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<uint64_t> result_vec1(1);
    result.visit([&](auto output) { result_vec1.assign(output.begin(), output.end()); });
    std::vector<uint64_t> result_vec2(1);
    // Identical calls should give different seeds every time with 1/(2^64) chance of a repeat.
    // We don't analyze for true randomness.
    result = p.eval({}).back();
    result.visit([&](auto output) { result_vec2.assign(output.begin(), output.end()); });
    EXPECT(result_vec1[0] != result_vec2[0]);
}
