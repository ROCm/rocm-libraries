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
#include <migraphx/op/reduce_sum.hpp>
#include <migraphx/op/reduce_mean.hpp>
#include <migraphx/op/reduce_max.hpp>
#include <migraphx/op/reduce_min.hpp>
#include <migraphx/op/reduce_prod.hpp>
#include <migraphx/op/reduce_all.hpp>
#include <migraphx/op/reduce_any.hpp>

template <class Op, int Axis, migraphx::shape::type_t T>
struct test_reduce_op_small : verify_program<test_reduce_op_small<Op, Axis, T>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{T, {3, 4, 2, 2}};
        auto x = mm->add_parameter("x", s);
        mm->add_instruction(Op{{Axis}}, x);
        return p;
    };

    std::string section() const { return "reduce"; }
};

template struct test_reduce_op_small<migraphx::op::reduce_sum, 1, migraphx::shape::float_type>;
template struct test_reduce_op_small<migraphx::op::reduce_sum, 3, migraphx::shape::float_type>;
template struct test_reduce_op_small<migraphx::op::reduce_sum, 2, migraphx::shape::int32_type>;
template struct test_reduce_op_small<migraphx::op::reduce_sum, 2, migraphx::shape::uint32_type>;
template struct test_reduce_op_small<migraphx::op::reduce_mean, 2, migraphx::shape::int32_type>;
template struct test_reduce_op_small<migraphx::op::reduce_max, 2, migraphx::shape::int32_type>;
template struct test_reduce_op_small<migraphx::op::reduce_min, 2, migraphx::shape::int32_type>;
template struct test_reduce_op_small<migraphx::op::reduce_all, 2, migraphx::shape::int32_type>;
template struct test_reduce_op_small<migraphx::op::reduce_any, 2, migraphx::shape::int32_type>;

template struct test_reduce_op_small<migraphx::op::reduce_sum, 3, migraphx::shape::half_type>;
template struct test_reduce_op_small<migraphx::op::reduce_sum, 2, migraphx::shape::half_type>;
template struct test_reduce_op_small<migraphx::op::reduce_mean, 2, migraphx::shape::half_type>;
template struct test_reduce_op_small<migraphx::op::reduce_max, 2, migraphx::shape::half_type>;
template struct test_reduce_op_small<migraphx::op::reduce_min, 2, migraphx::shape::half_type>;
template struct test_reduce_op_small<migraphx::op::reduce_prod, -2, migraphx::shape::half_type>;
template struct test_reduce_op_small<migraphx::op::reduce_all, 2, migraphx::shape::half_type>;
template struct test_reduce_op_small<migraphx::op::reduce_any, 2, migraphx::shape::half_type>;

template struct test_reduce_op_small<migraphx::op::reduce_sum, 3, migraphx::shape::bf16_type>;
template struct test_reduce_op_small<migraphx::op::reduce_sum, 2, migraphx::shape::bf16_type>;
template struct test_reduce_op_small<migraphx::op::reduce_mean, 2, migraphx::shape::bf16_type>;
template struct test_reduce_op_small<migraphx::op::reduce_max, 2, migraphx::shape::bf16_type>;
template struct test_reduce_op_small<migraphx::op::reduce_min, 2, migraphx::shape::bf16_type>;
template struct test_reduce_op_small<migraphx::op::reduce_prod, -2, migraphx::shape::bf16_type>;
template struct test_reduce_op_small<migraphx::op::reduce_all, 2, migraphx::shape::bf16_type>;
template struct test_reduce_op_small<migraphx::op::reduce_any, 2, migraphx::shape::bf16_type>;

template struct test_reduce_op_small<migraphx::op::reduce_sum,
                                     2,
                                     migraphx::shape::fp8e4m3fnuz_type>;
template struct test_reduce_op_small<migraphx::op::reduce_sum,
                                     3,
                                     migraphx::shape::fp8e4m3fnuz_type>;
template struct test_reduce_op_small<migraphx::op::reduce_mean,
                                     2,
                                     migraphx::shape::fp8e4m3fnuz_type>;
template struct test_reduce_op_small<migraphx::op::reduce_max,
                                     2,
                                     migraphx::shape::fp8e4m3fnuz_type>;
template struct test_reduce_op_small<migraphx::op::reduce_min,
                                     2,
                                     migraphx::shape::fp8e4m3fnuz_type>;
template struct test_reduce_op_small<migraphx::op::reduce_prod,
                                     -2,
                                     migraphx::shape::fp8e4m3fnuz_type>;
template struct test_reduce_op_small<migraphx::op::reduce_all,
                                     2,
                                     migraphx::shape::fp8e4m3fnuz_type>;
template struct test_reduce_op_small<migraphx::op::reduce_any,
                                     2,
                                     migraphx::shape::fp8e4m3fnuz_type>;

template struct test_reduce_op_small<migraphx::op::reduce_sum,
                                     2,
                                     migraphx::shape::fp8e5m2fnuz_type>;
template struct test_reduce_op_small<migraphx::op::reduce_sum,
                                     3,
                                     migraphx::shape::fp8e5m2fnuz_type>;
template struct test_reduce_op_small<migraphx::op::reduce_mean,
                                     2,
                                     migraphx::shape::fp8e5m2fnuz_type>;
template struct test_reduce_op_small<migraphx::op::reduce_max,
                                     2,
                                     migraphx::shape::fp8e5m2fnuz_type>;
template struct test_reduce_op_small<migraphx::op::reduce_min,
                                     2,
                                     migraphx::shape::fp8e5m2fnuz_type>;
template struct test_reduce_op_small<migraphx::op::reduce_prod,
                                     -2,
                                     migraphx::shape::fp8e5m2fnuz_type>;
template struct test_reduce_op_small<migraphx::op::reduce_all,
                                     2,
                                     migraphx::shape::fp8e5m2fnuz_type>;
template struct test_reduce_op_small<migraphx::op::reduce_any,
                                     2,
                                     migraphx::shape::fp8e5m2fnuz_type>;

template struct test_reduce_op_small<migraphx::op::reduce_sum, 2, migraphx::shape::fp8e4m3fn_type>;
template struct test_reduce_op_small<migraphx::op::reduce_sum, 3, migraphx::shape::fp8e4m3fn_type>;
template struct test_reduce_op_small<migraphx::op::reduce_mean, 2, migraphx::shape::fp8e4m3fn_type>;
template struct test_reduce_op_small<migraphx::op::reduce_max, 2, migraphx::shape::fp8e4m3fn_type>;
template struct test_reduce_op_small<migraphx::op::reduce_min, 2, migraphx::shape::fp8e4m3fn_type>;
template struct test_reduce_op_small<migraphx::op::reduce_prod,
                                     -2,
                                     migraphx::shape::fp8e4m3fn_type>;
template struct test_reduce_op_small<migraphx::op::reduce_all, 2, migraphx::shape::fp8e4m3fn_type>;
template struct test_reduce_op_small<migraphx::op::reduce_any, 2, migraphx::shape::fp8e4m3fn_type>;

template struct test_reduce_op_small<migraphx::op::reduce_sum, 2, migraphx::shape::fp8e5m2_type>;
template struct test_reduce_op_small<migraphx::op::reduce_sum, 3, migraphx::shape::fp8e5m2_type>;
template struct test_reduce_op_small<migraphx::op::reduce_mean, 2, migraphx::shape::fp8e5m2_type>;
template struct test_reduce_op_small<migraphx::op::reduce_max, 2, migraphx::shape::fp8e5m2_type>;
template struct test_reduce_op_small<migraphx::op::reduce_min, 2, migraphx::shape::fp8e5m2_type>;
template struct test_reduce_op_small<migraphx::op::reduce_prod, -2, migraphx::shape::fp8e5m2_type>;
template struct test_reduce_op_small<migraphx::op::reduce_all, 2, migraphx::shape::fp8e5m2_type>;
template struct test_reduce_op_small<migraphx::op::reduce_any, 2, migraphx::shape::fp8e5m2_type>;
