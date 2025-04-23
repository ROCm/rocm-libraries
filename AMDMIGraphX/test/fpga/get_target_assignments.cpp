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
#include "test.hpp"

#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/target_assignments.hpp>
#include <migraphx/iterator_for.hpp>

static migraphx::program create_program()
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto x    = mm->add_parameter("x", s);
    auto y    = mm->add_parameter("y", s);
    auto z    = mm->add_parameter("z", s);
    auto diff = mm->add_instruction(migraphx::make_op("add"), x, y);
    mm->add_instruction(migraphx::make_op("add"), diff, z);
    return p;
}

TEST_CASE(is_supported)
{
    auto p       = create_program();
    auto targets = migraphx::get_targets();
    EXPECT(not targets.empty());
    auto t = migraphx::make_target("fpga");

    const auto assignments = p.get_target_assignments({t});
    const auto* mod        = p.get_main_module();
    EXPECT(mod->size() == assignments.size());

    for(const auto ins : iterator_for(*mod))
    {
        const auto& target = assignments.at(ins);
        EXPECT(target == "fpga");
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
