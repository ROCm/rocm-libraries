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

TEST_CASE(gelu_fast_bias_test)
{
    migraphx::program p;
    auto type       = migraphx::shape::half_type;
    auto lens       = {3, 3};
    auto shape      = migraphx::shape{type, lens};
    auto* mm        = p.get_main_module();
    auto x          = mm->add_parameter("x", shape);
    auto bias       = mm->add_parameter("y", shape);
    x               = add_common_op(*mm, migraphx::make_op("add"), {x, bias});
    auto fit_const  = mm->add_literal(migraphx::literal{migraphx::shape{type}, {0.035677f}});
    auto sqrt_2_rpi = mm->add_literal(migraphx::literal{migraphx::shape{type}, {0.797885}});
    auto one        = mm->add_literal(migraphx::literal{migraphx::shape{type}, {1.0f}});
    auto half       = mm->add_literal(migraphx::literal{migraphx::shape{type}, {0.5f}});
    auto three      = mm->add_literal(migraphx::literal{migraphx::shape{type}, {3.0f}});

    auto pow0    = add_common_op(*mm, migraphx::make_op("pow"), {x, three});
    auto mul0    = add_common_op(*mm, migraphx::make_op("mul"), {pow0, fit_const});
    auto mul1    = add_common_op(*mm, migraphx::make_op("mul"), {sqrt_2_rpi, x});
    auto tanh_in = add_common_op(*mm, migraphx::make_op("add"), {mul0, mul1});
    auto tanh0   = mm->add_instruction(migraphx::make_op("tanh"), tanh_in);
    auto add1    = add_common_op(*mm, migraphx::make_op("add"), {tanh0, one});
    auto mul2    = add_common_op(*mm, migraphx::make_op("mul"), {x, half});
    add_common_op(*mm, migraphx::make_op("mul"), {add1, mul2});

    auto prog = optimize_onnx("gelu_fast_bias_test.onnx");

    EXPECT(p == prog);
}
