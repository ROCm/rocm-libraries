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
#include <migraphx/op/reduce_max.hpp>
#include <migraphx/op/reduce_mean.hpp>
#include <migraphx/op/reduce_min.hpp>
#include <migraphx/op/reduce_prod.hpp>
#include <migraphx/op/reduce_sum.hpp>
#include <migraphx/op/reduce_all.hpp>
#include <migraphx/op/reduce_any.hpp>

template <class Op, int Axis, migraphx::shape::type_t T>
struct test_reduce_op_large : verify_program<test_reduce_op_large<Op, Axis, T>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{T, {3, 1026, 4, 3}};
        auto x = mm->add_parameter("x", s);
        mm->add_instruction(Op{{Axis}}, x);
        return p;
    };

    std::string section() const { return "reduce"; }
};

template struct test_reduce_op_large<migraphx::op::reduce_max, 1, migraphx::shape::float_type>;
template struct test_reduce_op_large<migraphx::op::reduce_mean, 1, migraphx::shape::float_type>;
template struct test_reduce_op_large<migraphx::op::reduce_min, 1, migraphx::shape::float_type>;
template struct test_reduce_op_large<migraphx::op::reduce_prod, 2, migraphx::shape::float_type>;
template struct test_reduce_op_large<migraphx::op::reduce_sum, 1, migraphx::shape::float_type>;
template struct test_reduce_op_large<migraphx::op::reduce_all, 1, migraphx::shape::float_type>;
template struct test_reduce_op_large<migraphx::op::reduce_any, 1, migraphx::shape::float_type>;

template struct test_reduce_op_large<migraphx::op::reduce_all, 1, migraphx::shape::bool_type>;
template struct test_reduce_op_large<migraphx::op::reduce_any, 1, migraphx::shape::bool_type>;

template struct test_reduce_op_large<migraphx::op::reduce_sum,
                                     1,
                                     migraphx::shape::fp8e4m3fnuz_type>;
template struct test_reduce_op_large<migraphx::op::reduce_sum,
                                     1,
                                     migraphx::shape::fp8e5m2fnuz_type>;
template struct test_reduce_op_large<migraphx::op::reduce_sum, 1, migraphx::shape::fp8e4m3fn_type>;
template struct test_reduce_op_large<migraphx::op::reduce_sum, 1, migraphx::shape::fp8e5m2_type>;

struct test_reduce_mean_1 : verify_program<test_reduce_mean_1>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {1, 384, 1024}};
        auto x = mm->add_parameter("x", s);
        mm->add_instruction(migraphx::op::reduce_mean{{1}}, x);
        return p;
    };
};

struct test_reduce_mean_2 : verify_program<test_reduce_mean_2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {336, 400}};
        auto x = mm->add_parameter("x", s);
        mm->add_instruction(migraphx::op::reduce_mean{{1}}, x);
        return p;
    };
};

struct test_large_reduce_mean1 : verify_program<test_large_reduce_mean1>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 256 * 256 * 16}};
        auto x = mm->add_parameter("x", s);
        mm->add_instruction(migraphx::op::reduce_mean{{1}}, x);
        return p;
    };
};

struct test_large_reduce_mean2 : verify_program<test_large_reduce_mean2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {1, 32, 262144}};
        auto x = mm->add_parameter("x", s);
        mm->add_instruction(migraphx::op::reduce_mean{{2}}, x);
        return p;
    };
};
