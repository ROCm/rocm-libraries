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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

template <typename CType>
struct test_pow : verify_program<test_pow<CType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape::type_t dtype = migraphx::shape::get_type<CType>();
        auto* mm                      = p.get_main_module();
        migraphx::shape s{dtype, {6}};
        std::vector<float> vec_e(s.elements(), 2.0f);
        auto b = mm->add_parameter("x", s);
        auto e = mm->add_literal(migraphx::literal(s, vec_e));
        mm->add_instruction(migraphx::make_op("pow"), b, e);
        return p;
    }
};
template struct test_pow<float>;
template struct test_pow<migraphx::half>;
template struct test_pow<migraphx::bf16>;
template struct test_pow<migraphx::fp8::fp8e4m3fnuz>;
template struct test_pow<migraphx::fp8::fp8e5m2fnuz>;
template struct test_pow<migraphx::fp8::fp8e4m3fn>;
template struct test_pow<migraphx::fp8::fp8e5m2>;
