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
#include <migraphx/instruction.hpp>
#include <migraphx/common.hpp>

template <migraphx::shape::type_t DType>
struct test_conv_bn_add : verify_program<test_conv_bn_add<DType>>
{
    static migraphx::instruction_ref add_bn(migraphx::module& m, migraphx::instruction_ref x)
    {
        auto bn_lens = x->get_shape().lens();
        auto c_len   = bn_lens.at(1);

        migraphx::shape vars{DType, {c_len}};
        auto scale    = m.add_literal(migraphx::abs(migraphx::generate_literal(vars, 1 + c_len)));
        auto bias     = m.add_literal(migraphx::abs(migraphx::generate_literal(vars, 2 + c_len)));
        auto mean     = m.add_literal(migraphx::abs(migraphx::generate_literal(vars, 3 + c_len)));
        auto variance = m.add_literal(migraphx::abs(migraphx::generate_literal(vars, 4 + c_len)));

        auto rt  = m.add_literal(migraphx::literal{DType, {0.5}});
        auto eps = m.add_literal(migraphx::literal{DType, {1e-5f}});
        if constexpr(DType == migraphx::shape::fp8e4m3fnuz_type or
                     DType == migraphx::shape::fp8e4m3fn_type or
                     DType == migraphx::shape::fp8e5m2_type)
        {
            // use 0.250 for fp8
            eps = m.add_literal(migraphx::literal{DType, {0.250}});
        }
        auto usq_scale =
            m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), scale);
        auto usq_bias = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), bias);
        auto usq_mean = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), mean);
        auto usq_var =
            m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), variance);

        auto numer   = add_common_op(m, migraphx::make_op("sub"), {x, usq_mean});
        auto var_eps = add_common_op(m, migraphx::make_op("add"), {usq_var, eps});
        auto denom   = add_common_op(m, migraphx::make_op("pow"), {var_eps, rt});
        auto div0    = add_common_op(m, migraphx::make_op("div"), {numer, denom});
        auto r0      = add_common_op(m, migraphx::make_op("mul"), {div0, usq_scale});
        return add_common_op(m, migraphx::make_op("add"), {r0, usq_bias});
    }

    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm              = p.get_main_module();
        std::size_t ichannels = 64;
        std::size_t ochannels = 256;
        auto x                = mm->add_parameter("x", {DType, {1, ichannels, 56, 56}});
        auto w =
            mm->add_literal(migraphx::generate_literal({DType, {ochannels, ichannels, 1, 1}}, 1));
        auto y = mm->add_parameter("y", {DType, {1, ichannels, 56, 56}});
        auto v =
            mm->add_literal(migraphx::generate_literal({DType, {ochannels, ichannels, 1, 1}}, 2));
        auto relu1 = mm->add_instruction(migraphx::make_op("relu"), x);
        auto conv1 = mm->add_instruction(migraphx::make_op("convolution"), relu1, w);
        auto bn1   = add_bn(*mm, conv1);
        auto relu2 = mm->add_instruction(migraphx::make_op("relu"), y);
        auto conv2 = mm->add_instruction(migraphx::make_op("convolution"), relu2, v);
        auto bn2   = add_bn(*mm, conv2);
        auto sum   = mm->add_instruction(migraphx::make_op("add"), bn1, bn2);
        mm->add_instruction(migraphx::make_op("relu"), sum);
        return p;
    }
    std::string section() const { return "conv"; }
};

template struct test_conv_bn_add<migraphx::shape::float_type>;
template struct test_conv_bn_add<migraphx::shape::bf16_type>;
template struct test_conv_bn_add<migraphx::shape::fp8e4m3fnuz_type>;
template struct test_conv_bn_add<migraphx::shape::fp8e5m2fnuz_type>;
template struct test_conv_bn_add<migraphx::shape::fp8e4m3fn_type>;
template struct test_conv_bn_add<migraphx::shape::fp8e5m2_type>;
