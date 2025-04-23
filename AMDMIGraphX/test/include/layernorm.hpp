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
#include <vector>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

inline migraphx::instruction_ref add_layernorm(migraphx::module& m,
                                               migraphx::instruction_ref x,
                                               const std::vector<size_t>& dims,
                                               float eps = 1e-12f)
{
    auto mgx_type = x->get_shape().type();
    auto scale    = m.add_parameter("scale", migraphx::shape{mgx_type, {dims.back()}});
    auto bias     = m.add_parameter("bias", migraphx::shape{mgx_type, {dims.back()}});

    auto epsilon  = m.add_literal(migraphx::literal{migraphx::shape{mgx_type}, {eps}});
    auto exponent = m.add_literal(migraphx::literal{migraphx::shape{mgx_type}, {2.0f}});

    auto mean = m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), x);
    auto mean_mbcast =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", dims}}), mean);
    auto sub = m.add_instruction(migraphx::make_op("sub"), x, mean_mbcast);
    auto exponent_mbcast =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", dims}}), exponent);
    auto pow            = m.add_instruction(migraphx::make_op("pow"), sub, exponent_mbcast);
    auto var            = m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), pow);
    auto epsilon_mbcast = m.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", var->get_shape().lens()}}), epsilon);

    auto add_epsilon = m.add_instruction(migraphx::make_op("add"), var, epsilon_mbcast);
    auto sqrt        = m.add_instruction(migraphx::make_op("sqrt"), add_epsilon);
    auto sqrt_mbcast =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", dims}}), sqrt);
    auto div = m.add_instruction(migraphx::make_op("div"), sub, sqrt_mbcast);
    auto scale_mbcast =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", dims}}), scale);
    auto mul = m.add_instruction(migraphx::make_op("mul"), div, scale_mbcast);

    auto bias_mbcast =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", dims}}), bias);
    return m.add_instruction(migraphx::make_op("add"), mul, bias_mbcast);
}

inline migraphx::instruction_ref add_pointwise_layernorm(migraphx::module& m,
                                                         migraphx::instruction_ref x,
                                                         const std::vector<size_t>& dims,
                                                         float eps = 1e-12f)
{
    auto mgx_type = x->get_shape().type();
    auto scale    = m.add_parameter("scale", migraphx::shape{mgx_type, {1, 1, dims.back()}});
    auto bias     = m.add_parameter("bias", migraphx::shape{mgx_type, {1, 1, dims.back()}});

    auto epsilon = m.add_literal(migraphx::literal{migraphx::shape{mgx_type}, {eps}});
    auto one     = m.add_literal(migraphx::literal{migraphx::shape{mgx_type}, {1}});

    auto mean = m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), x);
    auto mean_mbcast =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", dims}}), mean);
    auto x_minus_mean = m.add_instruction(migraphx::make_op("sub"), x, mean_mbcast);
    auto sqdiff       = m.add_instruction(migraphx::make_op("sqdiff"), x, mean_mbcast);
    auto var = m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), sqdiff);

    auto epsilon_mbcast = m.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", var->get_shape().lens()}}), epsilon);
    auto var_stable   = m.add_instruction(migraphx::make_op("add"), var, epsilon_mbcast);
    auto inv_stddev_x = m.add_instruction(migraphx::make_op("rsqrt"), var_stable);
    auto inv_stddev_x_mbcast =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", dims}}), inv_stddev_x);
    auto norm = m.add_instruction(migraphx::make_op("mul"), x_minus_mean, inv_stddev_x_mbcast);

    auto one_mbcast = m.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", scale->get_shape().lens()}}), one);
    auto add_scale = m.add_instruction(migraphx::make_op("add"), scale, one_mbcast);
    auto scale_mbcast =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", dims}}), add_scale);
    auto scale_norm = m.add_instruction(migraphx::make_op("mul"), norm, scale_mbcast);

    auto bias_mbcast =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", dims}}), bias);

    return m.add_instruction(migraphx::make_op("add"), scale_norm, bias_mbcast);
}
