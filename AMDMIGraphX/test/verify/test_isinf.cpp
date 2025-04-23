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
#include <limits>
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

template <class T>
struct test_isinf : verify_program<test_isinf<T>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto max = std::numeric_limits<T>::max();
        auto min = std::numeric_limits<T>::min();
        auto inf = std::numeric_limits<T>::infinity();
        auto nan = std::numeric_limits<T>::quiet_NaN();
        auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::get_type<T>(), {5}});
        std::vector<T> data0{inf, -inf, max, min, nan};
        migraphx::shape s1{migraphx::shape::get_type<T>(), {5}};
        auto l0 = mm->add_literal(migraphx::literal{s1, data0});
        x       = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, l0);
        mm->add_instruction(migraphx::make_op("isinf"), x);
        return p;
    }
};

template struct test_isinf<migraphx::half>;
template struct test_isinf<migraphx::bf16>;
template struct test_isinf<float>;
