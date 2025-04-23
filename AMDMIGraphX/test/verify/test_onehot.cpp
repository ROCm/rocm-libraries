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
#include <migraphx/make_op.hpp>

template <migraphx::shape::type_t DType>
struct test_onehot : verify_program<test_onehot<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape inds_s{migraphx::shape::int64_type, {4}};
        migraphx::shape depth_s{migraphx::shape::int64_type, {1}, {0}};
        migraphx::shape values_s{DType, {2}};
        auto inds_lit     = mm->add_literal(migraphx::literal{inds_s, {0, -8, -1, 5}});
        auto depth_lit    = mm->add_literal(migraphx::literal{depth_s, {3}});
        auto values_param = mm->add_parameter("values", values_s);
        mm->add_instruction(
            migraphx::make_op("onehot", {{"axis", -1}}), inds_lit, depth_lit, values_param);
        return p;
    }
};
template struct test_onehot<migraphx::shape::float_type>;
template struct test_onehot<migraphx::shape::int32_type>;

struct test_onehot_param_inds : verify_program<test_onehot_param_inds>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape inds_s{migraphx::shape::int64_type, {2, 2}};
        migraphx::shape depth_s{migraphx::shape::int64_type, {1}, {0}};
        migraphx::shape values_s{migraphx::shape::float_type, {2}};
        auto inds_param   = mm->add_parameter("indices", inds_s);
        auto depth_lit    = mm->add_literal(migraphx::literal{depth_s, {3}});
        auto values_param = mm->add_parameter("values", values_s);
        mm->add_instruction(
            migraphx::make_op("onehot", {{"axis", 0}}), inds_param, depth_lit, values_param);
        return p;
    }
};
