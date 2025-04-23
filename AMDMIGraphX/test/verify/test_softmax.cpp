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

template <int Axis, migraphx::shape::type_t T>
struct test_softmax : verify_program<test_softmax<Axis, T>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{T, {512, 4, 1067, 6}};
        auto param = mm->add_parameter("0", s);
        mm->add_instruction(migraphx::make_op("softmax", {{"axis", Axis}}), param);

        return p;
    }

    std::string section() const { return "reduce"; }
};

template struct test_softmax<0, migraphx::shape::float_type>;
template struct test_softmax<2, migraphx::shape::float_type>;

template struct test_softmax<0, migraphx::shape::half_type>;
template struct test_softmax<1, migraphx::shape::half_type>;
template struct test_softmax<2, migraphx::shape::half_type>;
template struct test_softmax<3, migraphx::shape::half_type>;

template struct test_softmax<0, migraphx::shape::bf16_type>;
template struct test_softmax<1, migraphx::shape::bf16_type>;
template struct test_softmax<2, migraphx::shape::bf16_type>;
template struct test_softmax<3, migraphx::shape::bf16_type>;

template struct test_softmax<1, migraphx::shape::fp8e4m3fnuz_type>;
template struct test_softmax<3, migraphx::shape::fp8e4m3fnuz_type>;

template struct test_softmax<1, migraphx::shape::fp8e5m2fnuz_type>;
template struct test_softmax<3, migraphx::shape::fp8e5m2fnuz_type>;

template struct test_softmax<1, migraphx::shape::fp8e4m3fn_type>;
template struct test_softmax<3, migraphx::shape::fp8e4m3fn_type>;

template struct test_softmax<1, migraphx::shape::fp8e5m2_type>;
template struct test_softmax<3, migraphx::shape::fp8e5m2_type>;
