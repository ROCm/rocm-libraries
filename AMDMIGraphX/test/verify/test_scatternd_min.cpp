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
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_scatternd_min : verify_program<test_scatternd_min>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto dtype = migraphx::shape::float_type;
        auto itype = migraphx::shape::int64_type;
        migraphx::shape ds{dtype, {8}};
        migraphx::shape is{itype, {4, 1}};
        migraphx::shape us{dtype, {4}};
        std::vector<int64_t> ind_vec{4, 3, 1, 7};

        auto data    = mm->add_parameter("data", ds);
        auto indices = mm->add_literal(migraphx::literal{is, ind_vec});
        auto updates = mm->add_parameter("update", us);
        auto scatternd =
            mm->add_instruction(migraphx::make_op("scatternd_min"), data, indices, updates);
        mm->add_return({scatternd});

        return p;
    }
};
