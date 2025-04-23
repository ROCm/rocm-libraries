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
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/common.hpp>
#include <random>

template <migraphx::shape::type_t DType, unsigned int K, unsigned int N>
struct test_topk : verify_program<test_topk<DType, K, N>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm           = p.get_main_module();
        unsigned int batch = 3;
        migraphx::shape s{DType, {batch, N}};
        auto x1 = mm->add_parameter("x1", s);
        auto r  = mm->add_instruction(
            migraphx::make_op("topk", {{"axis", 1}, {"k", K}, {"largest", 1}}), x1);
        auto values  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), r);
        auto indices = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), r);
        mm->add_return({values, indices});

        return p;
    }

    std::size_t get_tolerance() const { return 2; };
};

template struct test_topk<migraphx::shape::half_type, 3, 9>;
template struct test_topk<migraphx::shape::half_type, 3, 84>;
template struct test_topk<migraphx::shape::half_type, 3, 240>;
template struct test_topk<migraphx::shape::half_type, 30, 840>;
template struct test_topk<migraphx::shape::half_type, 30, 2400>;
template struct test_topk<migraphx::shape::half_type, 100, 187>;
template struct test_topk<migraphx::shape::half_type, 100, 451>;
template struct test_topk<migraphx::shape::half_type, 100, 750>;
template struct test_topk<migraphx::shape::half_type, 100, 80000>;
template struct test_topk<migraphx::shape::half_type, 507, 507>;
template struct test_topk<migraphx::shape::half_type, 300, 24000>;
template struct test_topk<migraphx::shape::half_type, 1000, 1875>;
template struct test_topk<migraphx::shape::half_type, 1000, 30000>;
template struct test_topk<migraphx::shape::half_type, 1000, 120000>;
template struct test_topk<migraphx::shape::half_type, 1031, 1033>;

template struct test_topk<migraphx::shape::float_type, 30, 2400>;
template struct test_topk<migraphx::shape::float_type, 100, 80000>;
template struct test_topk<migraphx::shape::float_type, 1000, 1875>;
template struct test_topk<migraphx::shape::float_type, 1000, 120000>;
template struct test_topk<migraphx::shape::float_type, 1029, 80000>;
