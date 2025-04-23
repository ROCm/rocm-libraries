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
#include <migraphx/program.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_target.hpp>
#include "test.hpp"

TEST_CASE(perf_report)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::stringstream ss;
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(migraphx::make_op("add"), one, two);
    p.compile(migraphx::make_target("ref"));
    p.perf_report(ss, 2, {});

    std::string output = ss.str();
    EXPECT(migraphx::contains(output, "Summary:"));
    EXPECT(migraphx::contains(output, "Batch size:"));
    EXPECT(migraphx::contains(output, "Rate:"));
    EXPECT(migraphx::contains(output, "Total time:"));
    EXPECT(migraphx::contains(output, "Total instructions time:"));
    EXPECT(migraphx::contains(output, "Overhead time:"));
    EXPECT(migraphx::contains(output, "Overhead:"));
    EXPECT(not migraphx::contains(output, "fast"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
