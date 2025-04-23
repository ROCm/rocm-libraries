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
#include <migraphx/op/common.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/common.hpp>

template <migraphx::shape::type_t DType>
struct test_conv_bn_relu_pooling : verify_program<test_conv_bn_relu_pooling<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape xs{DType, {1, 3, 224, 224}};
        migraphx::shape ws{DType, {64, 3, 7, 7}};
        migraphx::shape vars{DType, {64}};
        auto x    = mm->add_parameter("x", xs);
        auto w    = mm->add_parameter("w", ws);
        auto conv = mm->add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {3, 3}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        auto scale    = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 1)));
        auto bias     = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 2)));
        auto mean     = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 3)));
        auto variance = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 4)));

        auto rt  = mm->add_literal(migraphx::literal{DType, {0.5}});
        auto eps = mm->add_literal(migraphx::literal{DType, {1e-5f}});
        if constexpr(DType == migraphx::shape::fp8e4m3fnuz_type or
                     DType == migraphx::shape::fp8e4m3fn_type or
                     DType == migraphx::shape::fp8e5m2_type)
        {
            // use 0.250 for fp8
            eps = mm->add_literal(migraphx::literal{DType, {0.250}});
        }
        auto usq_scale =
            mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), scale);
        auto usq_bias =
            mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), bias);
        auto usq_mean =
            mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), mean);
        auto usq_var =
            mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), variance);

        auto numer   = add_common_op(*mm, migraphx::make_op("sub"), {conv, usq_mean});
        auto var_eps = add_common_op(*mm, migraphx::make_op("add"), {usq_var, eps});
        auto denom   = add_common_op(*mm, migraphx::make_op("pow"), {var_eps, rt});
        auto div0    = add_common_op(*mm, migraphx::make_op("div"), {numer, denom});
        auto r0      = add_common_op(*mm, migraphx::make_op("mul"), {div0, usq_scale});
        auto bn      = add_common_op(*mm, migraphx::make_op("add"), {r0, usq_bias});

        auto relu = mm->add_instruction(migraphx::make_op("relu"), bn);
        mm->add_instruction(migraphx::make_op("pooling",
                                              {{"mode", migraphx::op::pooling_mode::average},
                                               {"padding", {1, 1}},
                                               {"stride", {2, 2}},
                                               {"lengths", {3, 3}},
                                               {"dilations", {1, 1}}}),
                            relu);
        return p;
    }
    std::string section() const { return "conv"; }
};

template struct test_conv_bn_relu_pooling<migraphx::shape::float_type>;
template struct test_conv_bn_relu_pooling<migraphx::shape::bf16_type>;
