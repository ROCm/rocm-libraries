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

#include <onnx_test.hpp>

TEST_CASE(selu_test)
{
    migraphx::program p;
    auto* mm                      = p.get_main_module();
    std::vector<std::size_t> lens = {2, 3};
    migraphx::shape s{migraphx::shape::double_type, lens};
    auto x = mm->add_parameter("x", s);

    migraphx::shape ls{migraphx::shape::double_type, {1}};
    auto la   = mm->add_literal({ls, {0.3}});
    auto lg   = mm->add_literal({ls, {0.25}});
    auto mbla = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), la);
    auto mblg = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), lg);

    auto sign_x = mm->add_instruction(migraphx::make_op("sign"), x);
    auto exp_x  = mm->add_instruction(migraphx::make_op("exp"), x);

    auto mlax  = mm->add_instruction(migraphx::make_op("mul"), mbla, exp_x);
    auto smlax = mm->add_instruction(migraphx::make_op("sub"), mlax, mbla);

    auto item1 = mm->add_instruction(migraphx::make_op("add"), smlax, x);
    auto item2 = mm->add_instruction(migraphx::make_op("sub"), smlax, x);

    auto sitem2 = mm->add_instruction(migraphx::make_op("mul"), sign_x, item2);
    auto item12 = mm->add_instruction(migraphx::make_op("sub"), item1, sitem2);
    auto r      = mm->add_instruction(migraphx::make_op("mul"), item12, mblg);
    mm->add_return({r});

    auto prog = read_onnx("selu_test.onnx");

    EXPECT(p == prog);
}
