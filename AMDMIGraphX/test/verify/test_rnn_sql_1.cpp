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
#include <migraphx/serialize.hpp>

#include <migraphx/make_op.hpp>

#include <migraphx/op/common.hpp>

template <migraphx::shape::type_t DType>
struct test_rnn_sql_1 : verify_program<test_rnn_sql_1<DType>>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 10;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape in_shape{DType, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{DType, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{DType, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{DType, {num_dirct, 2 * hidden_size}};
        migraphx::shape s_shape{migraphx::shape::int32_type, {batch_size}};
        migraphx::shape ih_shape{DType, {num_dirct, batch_size, hidden_size}};

        auto seq  = mm->add_parameter("seq", in_shape);
        auto w    = mm->add_parameter("w", w_shape);
        auto r    = mm->add_parameter("r", r_shape);
        auto bias = mm->add_parameter("bias", b_shape);
        std::vector<int> sl_data{5, 7};
        auto sql = mm->add_literal(migraphx::literal{s_shape, sl_data});
        auto ih  = mm->add_parameter("ih", ih_shape);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih);
        auto last_hs = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        mm->add_return({hs, last_hs});

        return p;
    }
    std::string section() const { return "rnn"; }
};

template struct test_rnn_sql_1<migraphx::shape::float_type>;
template struct test_rnn_sql_1<migraphx::shape::half_type>;
template struct test_rnn_sql_1<migraphx::shape::bf16_type>;
template struct test_rnn_sql_1<migraphx::shape::fp8e4m3fnuz_type>;
template struct test_rnn_sql_1<migraphx::shape::fp8e5m2fnuz_type>;
template struct test_rnn_sql_1<migraphx::shape::fp8e4m3fn_type>;
template struct test_rnn_sql_1<migraphx::shape::fp8e5m2_type>;
