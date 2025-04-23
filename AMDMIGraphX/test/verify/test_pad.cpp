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

template <migraphx::shape::type_t DType>
struct test_pad : verify_program<test_pad<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s0{DType, {1, 96, 165, 165}};
        std::vector<int64_t> pads0 = {0, 0, 0, 0, 0, 0, 1, 1};
        std::vector<int64_t> pads1 = {0, 0, 0, 0, 1, 1, 1, 1};
        std::vector<int64_t> pads2 = {1, 1, 1, 1, 0, 0, 0, 0};
        std::vector<int64_t> pads3 = {1, 0, 1, 0, 1, 0, 2, 0};
        auto l0                    = mm->add_parameter("x", s0);
        mm->add_instruction(migraphx::make_op("pad", {{"pads", pads0}}), l0);
        mm->add_instruction(migraphx::make_op("pad", {{"pads", pads1}}), l0);
        mm->add_instruction(migraphx::make_op("pad", {{"pads", pads2}}), l0);
        mm->add_instruction(migraphx::make_op("pad", {{"pads", pads3}}), l0);
        return p;
    }
};

template struct test_pad<migraphx::shape::int32_type>;
template struct test_pad<migraphx::shape::float_type>;
template struct test_pad<migraphx::shape::half_type>;
template struct test_pad<migraphx::shape::bf16_type>;
