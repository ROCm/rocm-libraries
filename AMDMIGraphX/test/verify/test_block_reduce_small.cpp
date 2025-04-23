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
#include <migraphx/common.hpp>
#include <migraphx/make_op.hpp>

template <int N, migraphx::shape::type_t T>
struct test_block_reduce_small : verify_program<test_block_reduce_small<N, T>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{T, {2, N}};
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto two = mm->add_literal(migraphx::literal{migraphx::shape{s.type(), {1}}, {2}});
        auto mul = migraphx::add_common_op(*mm, migraphx::make_op("div"), {x, two});
        auto r   = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), mul);
        auto rb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), r);
        auto add = mm->add_instruction(migraphx::make_op("add"), rb, y);
        mm->add_return({add});
        return p;
    };

    std::string section() const { return "reduce"; }
};

template <int N>
struct test_block_reduce_small_for_size : test_block_reduce_small<N, migraphx::shape::half_type>,
                                          test_block_reduce_small<N, migraphx::shape::bf16_type>,
                                          test_block_reduce_small<N, migraphx::shape::float_type>,
                                          test_block_reduce_small<N, migraphx::shape::int8_type>
{
};

template struct test_block_reduce_small_for_size<2>;
template struct test_block_reduce_small_for_size<3>;
template struct test_block_reduce_small_for_size<4>;
template struct test_block_reduce_small_for_size<8>;
template struct test_block_reduce_small_for_size<16>;
template struct test_block_reduce_small_for_size<25>;
template struct test_block_reduce_small_for_size<32>;
template struct test_block_reduce_small_for_size<64>;
template struct test_block_reduce_small_for_size<67>;
template struct test_block_reduce_small_for_size<128>;
template struct test_block_reduce_small_for_size<129>;
