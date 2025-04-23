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
#include <migraphx/marker.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/register_target.hpp>
#include "test.hpp"

struct mock_marker
{
    std::shared_ptr<std::stringstream> ss = std::make_shared<std::stringstream>();

    void mark_start(migraphx::instruction_ref ins_ref)
    {
        std::string text = "Mock marker instruction start:" + ins_ref->name();
        (*ss) << text;
    }
    void mark_stop(migraphx::instruction_ref)
    {
        std::string text = "Mock marker instruction stop.";
        (*ss) << text;
    }
    void mark_start(const migraphx::program&)
    {
        std::string text = "Mock marker program start.";
        (*ss) << text;
    }
    void mark_stop(const migraphx::program&)
    {
        std::string text = "Mock marker program stop.";
        (*ss) << text;
    }
};

TEST_CASE(marker)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(migraphx::make_op("add"), one, two);
    p.compile(migraphx::make_target("ref"));

    mock_marker temp_marker;
    p.mark({}, temp_marker);

    std::string output = temp_marker.ss->str();
    EXPECT(migraphx::contains(output, "Mock marker instruction start:@literal"));
    EXPECT(migraphx::contains(output, "Mock marker instruction start:ref::op"));
    EXPECT(migraphx::contains(output, "Mock marker instruction stop."));
    EXPECT(migraphx::contains(output, "Mock marker program start."));
    EXPECT(migraphx::contains(output, "Mock marker program stop."));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
