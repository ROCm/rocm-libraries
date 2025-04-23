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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/serialize.hpp>

#include <migraphx/make_op.hpp>

#include <migraphx/op/common.hpp>

struct test_lstm_forward_last_layout : verify_program<test_lstm_forward_last_layout>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape in_shape{migraphx::shape::float_type, {batch_size, seq_len, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {batch_size, num_dirct, hidden_size}};
        migraphx::shape l_shape{migraphx::shape::int32_type, {batch_size}};
        migraphx::shape ic_shape{migraphx::shape::float_type, {batch_size, num_dirct, hidden_size}};
        migraphx::shape pph_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size}};

        auto seq  = mm->add_parameter("seq", in_shape);
        auto w    = mm->add_parameter("w", w_shape);
        auto r    = mm->add_parameter("r", r_shape);
        auto bias = mm->add_parameter("bias", b_shape);
        auto ih   = mm->add_parameter("ih", ih_shape);
        auto len  = mm->add_literal(migraphx::literal(l_shape, {1, 2}));
        auto ic   = mm->add_parameter("ic", ic_shape);
        auto pph  = mm->add_parameter("pph", pph_shape);

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);
        ic  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ic);

        auto output = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            len,
            ih,
            ic,
            pph);
        auto last_output =
            mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), output, len);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_output);

        return p;
    }
    std::string section() const { return "rnn"; }
};
