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
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/common.hpp>
#include <migraphx/make_op.hpp>

static migraphx::instruction_ref add_instancenorm(migraphx::module& m,
                                                  migraphx::instruction_ref x,
                                                  const std::vector<size_t>& dims,
                                                  float eps = 1e-5f)
{
    auto mgx_type = x->get_shape().type();
    auto x_lens   = x->get_shape().lens();
    std::vector<size_t> axes(x_lens.size() - 2);
    std::iota(axes.begin(), axes.end(), 2);
    auto scale   = m.add_parameter("scale", migraphx::shape{mgx_type, dims});
    auto bias    = m.add_parameter("bias", migraphx::shape{mgx_type, dims});
    auto epsilon = m.add_literal(migraphx::literal{migraphx::shape{mgx_type}, {eps}});

    auto mean = m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), x);
    auto mean_mbcast =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", x_lens}}), mean);
    auto sub = m.add_instruction(migraphx::make_op("sub"), x, mean_mbcast);
    auto l0  = m.add_instruction(migraphx::make_op("sqdiff"), {x, mean_mbcast});
    auto var = m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), {l0});
    auto epsilon_mbcast =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", x_lens}}), epsilon);
    auto var_mbcast =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", x_lens}}), var);
    auto add_epsilon = m.add_instruction(migraphx::make_op("add"), var_mbcast, epsilon_mbcast);
    auto rsqrt       = m.add_instruction(migraphx::make_op("rsqrt"), add_epsilon);
    auto l1          = m.add_instruction(migraphx::make_op("mul"), {sub, rsqrt});
    auto scale_mbcast =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", x_lens}}), scale);
    auto mul = m.add_instruction(migraphx::make_op("mul"), scale_mbcast, l1);
    auto bias_mbcast =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", x_lens}}), bias);
    return m.add_instruction(migraphx::make_op("add"), mul, bias_mbcast);
}

template <migraphx::shape::type_t TYPE>
struct test_instancenorm : verify_program<test_instancenorm<TYPE>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                 = p.get_main_module();
        std::vector<size_t> dims = {1, 2, 5, 5};
        auto x                   = mm->add_parameter("x", migraphx::shape{TYPE, dims});
        add_instancenorm(*mm, x, {1, 2, 1, 1});
        return p;
    }

    std::string section() const { return "reduce"; }
};
template struct test_instancenorm<migraphx::shape::float_type>;
template struct test_instancenorm<migraphx::shape::half_type>;

template <migraphx::shape::type_t TYPE>
struct test_instancenorm_large_3d : verify_program<test_instancenorm_large_3d<TYPE>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                 = p.get_main_module();
        std::vector<size_t> dims = {1, 32, 64, 64, 64};
        auto x                   = mm->add_parameter("x", migraphx::shape{TYPE, dims});
        add_instancenorm(*mm, x, {1, 32, 1, 1, 1});
        return p;
    }

    std::string section() const { return "reduce"; }
};

template struct test_instancenorm_large_3d<migraphx::shape::float_type>;
template struct test_instancenorm_large_3d<migraphx::shape::half_type>;
template struct test_instancenorm_large_3d<migraphx::shape::bf16_type>;
