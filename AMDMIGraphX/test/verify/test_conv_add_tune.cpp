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

template <migraphx::shape::type_t DType>
struct test_conv_add_tune : verify_program<test_conv_add_tune<DType>>
{
    // this test is for testing MLIR split-k convolution perfConfigs and problemKey clash in problem
    // cache
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        // choose sizes such that, it would pick mlir for convolutions
        auto x1    = mm->add_parameter("x1", {DType, {1, 256, 16, 16}});
        auto w1    = mm->add_literal(migraphx::generate_literal({DType, {1, 256, 3, 2}}, 1));
        auto x2    = mm->add_parameter("x2", {DType, {1, 256, 16, 16}});
        auto w2    = mm->add_literal(migraphx::generate_literal({DType, {1, 256, 3, 2}}, 1));
        auto conv1 = mm->add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 0}}, {"stride", {2, 2}}}),
            x1,
            w1);
        // add pooling so that it doesn't get fused with conv1.
        auto pooling =
            mm->add_instruction(migraphx::make_op("pooling",
                                                  {{"mode", migraphx::op::pooling_mode::average},
                                                   {"padding", {1, 1, 1, 1}},
                                                   {"stride", {1, 1}},
                                                   {"lengths", {3, 3}},
                                                   {"count_include_pad", false}}),
                                conv1);
        // conv2 + pointwise-add
        auto conv2 = mm->add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 0}}, {"stride", {2, 2}}}),
            x2,
            w2);
        mm->add_instruction(migraphx::make_op("add"), pooling, conv2);
        return p;
    }
    // Turn on Exhaustive-tune to enable split-k perf-configs from MLIR
    migraphx::compile_options get_compile_options() const
    {
        return migraphx::compile_options{.exhaustive_tune = true};
    }
    std::string section() const { return "conv"; }
};

template struct test_conv_add_tune<migraphx::shape::float_type>;
template struct test_conv_add_tune<migraphx::shape::half_type>;
template struct test_conv_add_tune<migraphx::shape::bf16_type>;
template struct test_conv_add_tune<migraphx::shape::fp8e4m3fnuz_type>;
template struct test_conv_add_tune<migraphx::shape::fp8e5m2fnuz_type>;
template struct test_conv_add_tune<migraphx::shape::fp8e4m3fn_type>;
template struct test_conv_add_tune<migraphx::shape::fp8e5m2_type>;
