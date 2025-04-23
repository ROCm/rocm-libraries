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
#include <migraphx/serialize.hpp>

#include <migraphx/make_op.hpp>

template <migraphx::shape::type_t From, migraphx::shape::type_t To>
struct test_convert : verify_program<test_convert<From, To>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{From, {8, 24}};
        migraphx::shape sb{From, {24, 6}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);
        auto ia = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::to_value(To)}}), pa);
        auto ib = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::to_value(To)}}), pb);
        mm->add_instruction(migraphx::make_op("dot"), ia, ib);

        return p;
    };
    std::string section() const { return "gemm"; }
};

template struct test_convert<migraphx::shape::int8_type, migraphx::shape::float_type>;
template struct test_convert<migraphx::shape::fp8e4m3fnuz_type, migraphx::shape::float_type>;
template struct test_convert<migraphx::shape::fp8e5m2fnuz_type, migraphx::shape::float_type>;
template struct test_convert<migraphx::shape::fp8e4m3fn_type, migraphx::shape::float_type>;
template struct test_convert<migraphx::shape::fp8e5m2_type, migraphx::shape::float_type>;
template struct test_convert<migraphx::shape::half_type, migraphx::shape::float_type>;
template struct test_convert<migraphx::shape::bf16_type, migraphx::shape::float_type>;
