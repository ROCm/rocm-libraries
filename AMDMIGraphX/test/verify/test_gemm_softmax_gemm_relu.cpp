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

enum class bias
{
    without,
    with,
    with_standard_shape
};

template <bias Config>
struct test_gemm_softmax_gemm_relu : verify_program<test_gemm_softmax_gemm_relu<Config>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape m1_shape{migraphx::shape::half_type, {1, 12, 256, 256}};
        auto m2_elements = m1_shape.elements();
        auto a           = mm->add_parameter("1", m1_shape);
        auto b           = mm->add_parameter("2", m1_shape);
        auto b1          = mm->add_parameter("3", m1_shape);
        std::vector<float> eights(m2_elements, 0.125);
        auto eight = mm->add_literal(migraphx::literal{m1_shape, eights});

        b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), b);
        auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto scale = mm->add_instruction(migraphx::make_op("mul"), gemm1, eight);

        std::optional<migraphx::instruction_ref> add_bias{std::nullopt};
        if constexpr(Config == bias::with or Config == bias::with_standard_shape)
        {
            auto bias_shape = m1_shape;
            if(Config != bias::with_standard_shape)
            {
                bias_shape = migraphx::shape::from_permutation(
                    bias_shape.type(), bias_shape.lens(), {0, 1, 3, 2});
            }
            auto bias_term = mm->add_parameter("4", bias_shape);
            add_bias       = mm->add_instruction(migraphx::make_op("add"), scale, bias_term);
        }

        auto softmax = mm->add_instruction(migraphx::make_op("softmax", {{"axis", 3}}),
                                           Config == bias::without ? scale : add_bias.value());
        auto gemm2   = mm->add_instruction(migraphx::make_op("dot"), softmax, b1);
        mm->add_instruction(migraphx::make_op("relu"), gemm2);
        return p;
    }
    std::string section() const { return "gemm"; }
};

template struct test_gemm_softmax_gemm_relu<bias::without>;
template struct test_gemm_softmax_gemm_relu<bias::with>;
template struct test_gemm_softmax_gemm_relu<bias::with_standard_shape>;
