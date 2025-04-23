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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

template <typename Derived>
struct test_scatter_elements_duplicate_index_base : verify_program<Derived>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sd{migraphx::shape::float_type, {3, 3}};
        migraphx::shape si{migraphx::shape::int32_type, {3, 2}};
        std::vector<int> vi = {1, 0, 1, 0, 1, 0};
        migraphx::shape su{migraphx::shape::float_type, {3, 2}};

        auto pd              = mm->add_parameter("data", sd);
        auto li              = mm->add_literal(migraphx::literal{si, vi});
        auto pu              = mm->add_parameter("update", su);
        const auto reduction = static_cast<const Derived&>(*this).reduction();
        auto r               = mm->add_instruction(
            migraphx::make_op("scatter_" + reduction, {{"axis", 0}}), pd, li, pu);
        mm->add_return({r});

        return p;
    }
};

struct test_scatter_elements_add_duplicate_index
    : test_scatter_elements_duplicate_index_base<test_scatter_elements_add_duplicate_index>
{
    std::string reduction() const { return "add"; }
};

struct test_scatter_elements_mul_duplicate_index
    : test_scatter_elements_duplicate_index_base<test_scatter_elements_mul_duplicate_index>
{
    std::string reduction() const { return "mul"; }
};

struct test_scatter_elements_min_duplicate_index
    : test_scatter_elements_duplicate_index_base<test_scatter_elements_min_duplicate_index>
{
    std::string reduction() const { return "min"; }
};

struct test_scatter_elements_max_duplicate_index
    : test_scatter_elements_duplicate_index_base<test_scatter_elements_max_duplicate_index>
{
    std::string reduction() const { return "max"; }
};
