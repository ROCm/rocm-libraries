/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(allocate_dyn0)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int64_type, {4}};
    auto out_dims = mm->add_parameter("out_dims", s);
    mm->add_instruction(migraphx::make_op("allocate", {{"buf_type", migraphx::shape::float_type}}),
                        out_dims);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int64_t> data = {2, 3, 4, 4};
    params["out_dims"]        = migraphx::argument(s, data.data());
    auto result               = p.eval(params).back();
    migraphx::shape sresult{migraphx::shape::float_type, {2, 3, 4, 4}};
    result.visit([&](auto output) { EXPECT(output.get_shape() == sresult); });
}

TEST_CASE(allocate_dyn1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int64_type, {4}};
    migraphx::shape out_shape{migraphx::shape::float_type, {2, 3, 4, 4}};
    auto out_dims = mm->add_parameter("out_dims", s);
    mm->add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(out_shape)}}),
                        out_dims);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int64_t> data = {2, 3, 4, 4};
    params["out_dims"]        = migraphx::argument(s, data.data());
    auto result               = p.eval(params).back();
    result.visit([&](auto output) { EXPECT(output.get_shape() == out_shape); });
}
