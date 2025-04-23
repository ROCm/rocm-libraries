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
#include <migraphx/make_op.hpp>
#include <limits>
#include <type_traits>

template <migraphx::shape::type_t Q, typename T>
struct test_literal_limits : verify_program<test_literal_limits<Q, T>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto input_s = migraphx::shape(Q, {3, 1});
        T infinity_val{0};
        if constexpr(std::numeric_limits<T>::has_infinity and std::is_floating_point<T>{})
        {
            infinity_val = std::numeric_limits<T>::infinity();
        }
        std::vector<T> s_data{
            infinity_val, static_cast<T>(-infinity_val), std::numeric_limits<T>::quiet_NaN()};

        auto input_param = mm->add_parameter("test_input", input_s);
        auto input       = mm->add_literal(migraphx::literal{input_s, s_data});
        auto o1          = mm->add_instruction(migraphx::make_op("mul"), input, input_param);
        mm->add_instruction(migraphx::make_op("isnan"), o1);
        return p;
    }
};

template struct test_literal_limits<migraphx::shape::float_type, float>;
template struct test_literal_limits<migraphx::shape::double_type, double>;
template struct test_literal_limits<migraphx::shape::half_type, migraphx::half>;
template struct test_literal_limits<migraphx::shape::bf16_type, migraphx::bf16>;
template struct test_literal_limits<migraphx::shape::int32_type, int32_t>;
template struct test_literal_limits<migraphx::shape::int8_type, int8_t>;
template struct test_literal_limits<migraphx::shape::fp8e4m3fnuz_type, migraphx::fp8::fp8e4m3fnuz>;
template struct test_literal_limits<migraphx::shape::fp8e5m2fnuz_type, migraphx::fp8::fp8e5m2fnuz>;
template struct test_literal_limits<migraphx::shape::fp8e4m3fn_type, migraphx::fp8::fp8e4m3fn>;
template struct test_literal_limits<migraphx::shape::fp8e5m2_type, migraphx::fp8::fp8e5m2>;
