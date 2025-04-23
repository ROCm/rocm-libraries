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
#include <migraphx/common.hpp>

/*
    Checking for y == 0 ? eps : y

    Adding this because HIP fmod sign changes when y = 0 resulting in nan and -nan not beign
   consistent between ref and gpu implementations.
*/
static migraphx::instruction_ref
add_epsilon(migraphx::module& m,
            migraphx::instruction_ref y,
            migraphx::shape::type_t dtype = migraphx::shape::float_type)
{
    auto zero = m.add_literal(migraphx::literal{migraphx::shape{dtype}, {0.0f}});
    auto eps  = m.add_literal(migraphx::literal{migraphx::shape{dtype}, {1e-3f}});
    auto op_y = add_common_op(m, migraphx::make_op("equal"), {y, zero});
    return add_common_op(m, migraphx::make_op("where"), {op_y, eps, y});
}

template <migraphx::shape::type_t DType>
struct test_fmod : verify_program<test_fmod<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {64}};
        auto x        = mm->add_parameter("x", s);
        auto y        = mm->add_parameter("y", s);
        auto op_where = add_epsilon(*mm, y, DType);
        mm->add_instruction(migraphx::make_op("fmod"), x, op_where);
        return p;
    }
};
template struct test_fmod<migraphx::shape::float_type>;
template struct test_fmod<migraphx::shape::half_type>;
template struct test_fmod<migraphx::shape::bf16_type>;
template struct test_fmod<migraphx::shape::fp8e4m3fnuz_type>;
template struct test_fmod<migraphx::shape::fp8e4m3fn_type>;
template struct test_fmod<migraphx::shape::fp8e5m2_type>;

template <migraphx::shape::type_t DType>
struct test_mod : verify_program<test_mod<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {64}};
        auto x        = mm->add_parameter("x", s);
        auto y        = mm->add_parameter("y", s);
        auto op_where = add_epsilon(*mm, y, DType);
        mm->add_instruction(migraphx::make_op("mod"), x, op_where);
        return p;
    }
};

template struct test_mod<migraphx::shape::float_type>;
template struct test_mod<migraphx::shape::half_type>;
template struct test_mod<migraphx::shape::bf16_type>;
template struct test_mod<migraphx::shape::fp8e4m3fnuz_type>;
template struct test_mod<migraphx::shape::fp8e5m2fnuz_type>;
template struct test_mod<migraphx::shape::fp8e4m3fn_type>;
template struct test_mod<migraphx::shape::fp8e5m2_type>;
