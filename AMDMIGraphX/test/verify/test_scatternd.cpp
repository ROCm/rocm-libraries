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
#include <migraphx/shape.hpp>

template <migraphx::shape::type_t DType>
struct test_scatternd : verify_program<test_scatternd<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto itype = migraphx::shape::int64_type;
        migraphx::shape ds{DType, {1}};
        migraphx::shape is{itype, {4, 1}};
        migraphx::shape us{DType, {4}};
        std::vector<int64_t> ind_vec{4, 3, 1, 7};

        auto ld = mm->add_literal(migraphx::literal{ds, {1}});
        auto data =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8}}}), ld);
        auto indices = mm->add_literal(migraphx::literal{is, ind_vec});
        auto updates = mm->add_parameter("update", us);
        auto scatternd =
            mm->add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
        mm->add_return({scatternd});

        return p;
    }
};

template struct test_scatternd<migraphx::shape::float_type>;
template struct test_scatternd<migraphx::shape::half_type>;
template struct test_scatternd<migraphx::shape::bf16_type>;
template struct test_scatternd<migraphx::shape::fp8e4m3fnuz_type>;
