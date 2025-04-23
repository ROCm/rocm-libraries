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

template <class Derived, int64_t Axis, int64_t Direction, int64_t Idx>
struct test_scan_slice_base : verify_program<Derived>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape data_sh{migraphx::shape::int32_type, {2, 2, 2}};
        auto data_param = mm->add_parameter("data", data_sh);
        migraphx::shape idx_sh{migraphx::shape::int64_type, {1}};
        auto idx_lit = mm->add_literal(migraphx::literal{idx_sh, {Idx}});

        mm->add_instruction(
            migraphx::make_op("scan_slice", {{"axis", Axis}, {"direction", Direction}}),
            data_param,
            idx_lit);

        return p;
    }
};

struct test_scan_slice1 : test_scan_slice_base<test_scan_slice1, 0, 0, 0>
{
};

struct test_scan_slice2 : test_scan_slice_base<test_scan_slice2, -1, 1, 1>
{
};

struct test_scan_slice3 : test_scan_slice_base<test_scan_slice2, 1, 0, 1>
{
};
