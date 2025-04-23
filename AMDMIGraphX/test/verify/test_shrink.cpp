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
#include <migraphx/op/common.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/common.hpp>

template <migraphx::shape::type_t T>
struct test_shrink : verify_program<test_shrink<T>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        float bias  = 1.5;
        float lambd = 1.5;
        auto* mm    = p.get_main_module();
        migraphx::shape is{T, {2, 3}};
        std::vector<float> data;
        migraphx::shape::visit(T, [&](auto as) {
            as.is_signed() ? data.assign({-3.0, -2.0, -1.0, 0.0, 1.0, 2.0})
                           : data.assign({3.0, 2.0, 1.0, 0.0, 1.0, 2.0});
        });
        auto x        = mm->add_literal(migraphx::literal{is, data});
        auto lit_bias = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {bias}});
        auto lit_neg_lambd =
            mm->add_literal(migraphx::literal{migraphx::shape::float_type, {-lambd}});
        auto lit_lambd = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {lambd}});

        auto x_plus_bias = add_common_op(*mm, migraphx::make_op("add"), {x, lit_bias});
        auto x_min_bias  = add_common_op(*mm, migraphx::make_op("sub"), {x, lit_bias});

        auto cond1   = add_common_op(*mm, migraphx::make_op("less"), {x, lit_neg_lambd});
        auto cond2_a = add_common_op(*mm, migraphx::make_op("not"), {cond1});
        auto cond2_b = add_common_op(*mm, migraphx::make_op("greater"), {x, lit_lambd});
        auto cond2   = add_common_op(*mm, migraphx::make_op("logical_and"), {cond2_a, cond2_b});

        auto mul1 = mm->add_instruction(migraphx::make_op("convert", {{"target_type", T}}), cond1);
        auto mul2 = mm->add_instruction(migraphx::make_op("convert", {{"target_type", T}}), cond2);

        auto first  = add_common_op(*mm, migraphx::make_op("mul"), {mul1, x_plus_bias});
        auto second = add_common_op(*mm, migraphx::make_op("mul"), {mul2, x_min_bias});
        auto ret    = add_common_op(*mm, migraphx::make_op("add"), {first, second});
        if(ret->get_shape().type() != T)
        {
            mm->add_instruction(migraphx::make_op("convert", {{"target_type", T}}), ret);
        }
        return p;
    }
};

template struct test_shrink<migraphx::shape::double_type>;
template struct test_shrink<migraphx::shape::float_type>;
template struct test_shrink<migraphx::shape::half_type>;
template struct test_shrink<migraphx::shape::bf16_type>;
template struct test_shrink<migraphx::shape::int64_type>;
template struct test_shrink<migraphx::shape::int32_type>;
template struct test_shrink<migraphx::shape::int16_type>;
template struct test_shrink<migraphx::shape::int8_type>;
template struct test_shrink<migraphx::shape::uint64_type>;
template struct test_shrink<migraphx::shape::uint32_type>;
template struct test_shrink<migraphx::shape::uint16_type>;
template struct test_shrink<migraphx::shape::uint8_type>;
