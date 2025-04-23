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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include "test.hpp"

TEST_CASE(rnn_forward)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 2;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    std::vector<float> w_data{0.4691,
                              0.3185,
                              -0.2227,
                              0.4423,
                              -0.0609,
                              -0.2803,
                              0.1744,
                              0.3146,
                              0.4049,
                              -0.3973,
                              -0.0890,
                              -0.1636};

    std::vector<float> r_data{-0.0456,
                              0.1061,
                              0.1574,
                              -0.4928,
                              -0.4300,
                              -0.1909,
                              -0.0225,
                              -0.2668,
                              0.1840,
                              -0.4453,
                              -0.4896,
                              0.1302,
                              -0.0929,
                              0.3545,
                              -0.4981,
                              0.0616};

    std::vector<float> bias_data{
        -0.4938, 0.4355, -0.3186, 0.2094, 0.1037, -0.1071, 0.4504, -0.3990};
    std::vector<float> ih_data(num_dirct * batch_size * hidden_size, 0);
    std::vector<float> input(seq_len * batch_size * input_size, 0);
    input[0] = input[1] = 1.0;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
    float clip = 0.0f;
    // concatenation of hidden states as program output
    {

        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        mm->add_return({hs, lho});
        p.compile(migraphx::make_target("ref"));

        auto outputs = p.eval({});
        const auto& res_hs  = outputs.front();
        const auto& res_lho = outputs.back();

        std::vector<float> hs_data;
        std::vector<float> lho_data;
        res_hs.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });
        res_lho.visit([&](auto output) { lho_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{0.37780784,
                                        0.61055139,
                                        0.55168478,
                                        -0.5888475,
                                        -0.37144644,
                                        0.31708236,
                                        0.13104209,
                                        -0.18736027,
                                        0.03445704,
                                        0.19167931,
                                        -0.3946827,
                                        -0.30889652,
                                        -0.22276389,
                                        0.44193283,
                                        -0.16477929,
                                        -0.11893477};

        std::vector<float> lho_data_gold{0.03445704,
                                         0.19167931,
                                         -0.3946827,
                                         -0.30889652,
                                         -0.22276389,
                                         0.44193283,
                                         -0.16477929,
                                         -0.11893477};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(lho_data, lho_data_gold));
    }

    {
        migraphx::program p;
        auto* mm      = p.get_main_module();
        auto seq_orig = mm->add_literal(migraphx::literal{in_shape, input});
        auto ih       = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto w        = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r        = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias     = mm->add_literal(migraphx::literal{b_shape, bias_data});
        migraphx::shape pad_seq_s{migraphx::shape::float_type, {2, batch_size, input_size}};
        std::vector<float> pad_data(pad_seq_s.elements(), 0.0f);
        auto seq_p = mm->add_literal(migraphx::literal{pad_seq_s, pad_data});
        auto seq = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), seq_orig, seq_p);
        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data(batch_size, static_cast<int32_t>(seq_len));
        auto sql = mm->add_literal(seq_len_s, len_data);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func", {}},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih);
        auto last_out = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        mm->add_return({out_hs, last_out});
        p.compile(migraphx::make_target("ref"));

        auto outputs = p.eval({});

        const auto& arg_hs          = outputs.front();
        const auto& arg_last_output = outputs.back();
        std::vector<float> last_output_data;
        std::vector<float> hs_data;
        arg_hs.visit([&](auto out) { hs_data.assign(out.begin(), out.end()); });
        arg_last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> hs_data_gold{
            0.37780784,  0.61055139,  0.55168478, -0.5888475, -0.37144644, 0.31708236,  0.13104209,
            -0.18736027, 0.03445704,  0.19167931, -0.3946827, -0.30889652, -0.22276389, 0.44193283,
            -0.16477929, -0.11893477, 0.0,        0.0,        0.0,         0.0,         0.0,
            0.0,         0.0,         0.0,        0.0,        0.0,         0.0,         0.0,
            0.0,         0.0,         0.0,        0.0};

        std::vector<float> last_output_data_gold{0.03445704,
                                                 0.19167931,
                                                 -0.3946827,
                                                 -0.30889652,
                                                 -0.22276389,
                                                 0.44193283,
                                                 -0.16477929,
                                                 -0.11893477};
        EXPECT(migraphx::verify::verify_rms_range(last_output_data, last_output_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data{2, 1};
        auto sql = mm->add_literal(seq_len_s, len_data);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func", {}},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih);
        auto last_out = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        mm->add_return({out_hs, last_out});
        p.compile(migraphx::make_target("ref"));

        auto outputs = p.eval({});

        const auto& arg_hs          = outputs.front();
        const auto& arg_last_output = outputs.back();
        std::vector<float> last_output_data;
        std::vector<float> hs_data;
        arg_hs.visit([&](auto out) { hs_data.assign(out.begin(), out.end()); });
        arg_last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> hs_data_gold{0.377808,
                                        0.610551,
                                        0.551685,
                                        -0.588848,
                                        -0.371446,
                                        0.317082,
                                        0.131042,
                                        -0.18736,
                                        0.034457,
                                        0.191679,
                                        -0.394683,
                                        -0.308897,
                                        0,
                                        0,
                                        0,
                                        0};
        std::vector<float> last_output_data_gold{
            0.034457, 0.191679, -0.394683, -0.308897, -0.371446, 0.317082, 0.131042, -0.18736};
        EXPECT(migraphx::verify::verify_rms_range(last_output_data, last_output_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // 3 args
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input});
        auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func", {}},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            seq,
            w,
            r);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        p.compile(migraphx::make_target("ref"));

        auto last_output = p.eval({}).back();
        std::vector<float> last_output_data;
        last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> last_output_data_gold{
            0.2935145, -0.23719997, -0.31123261, -0.18357255, 0., 0., 0., 0.};
        EXPECT(migraphx::verify::verify_rms_range(last_output_data, last_output_data_gold));
    }

    // seq_len = 1
    {
        seq_len = 1;
        std::vector<float> input_1(seq_len * batch_size * input_size, 0);
        input_1[0] = input_1[1] = 1.0;
        migraphx::shape in_shape_1{migraphx::shape::float_type, {seq_len, batch_size, input_size}};

        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape_1, input_1});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        mm->add_instruction(
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
            und,
            ih);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{0.37780784,
                                        0.61055139,
                                        0.55168478,
                                        -0.5888475,
                                        -0.37144644,
                                        0.31708236,
                                        0.13104209,
                                        -0.18736027};
        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(rnn_forward_layout)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 2;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    std::vector<float> w_data{0.4691,
                              0.3185,
                              -0.2227,
                              0.4423,
                              -0.0609,
                              -0.2803,
                              0.1744,
                              0.3146,
                              0.4049,
                              -0.3973,
                              -0.0890,
                              -0.1636};

    std::vector<float> r_data{-0.0456,
                              0.1061,
                              0.1574,
                              -0.4928,
                              -0.4300,
                              -0.1909,
                              -0.0225,
                              -0.2668,
                              0.1840,
                              -0.4453,
                              -0.4896,
                              0.1302,
                              -0.0929,
                              0.3545,
                              -0.4981,
                              0.0616};

    std::vector<float> bias_data{
        -0.4938, 0.4355, -0.3186, 0.2094, 0.1037, -0.1071, 0.4504, -0.3990};
    std::vector<float> ih_data(num_dirct * batch_size * hidden_size, 0);
    std::vector<float> input(seq_len * batch_size * input_size, 0);
    input[0] = input[1] = 1.0;
    migraphx::shape in_shape{migraphx::shape::float_type, {batch_size, seq_len, input_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {batch_size, num_dirct, hidden_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
    float clip = 0.0f;
    // concatenation of hidden states as program output
    {

        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        hs  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}), hs);
        lho = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), lho);
        mm->add_return({hs, lho});
        p.compile(migraphx::make_target("ref"));

        auto outputs = p.eval({});
        const auto& res_hs  = outputs.front();
        const auto& res_lho = outputs.back();

        std::vector<float> hs_data;
        std::vector<float> lho_data;
        res_hs.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });
        res_lho.visit([&](auto output) { lho_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{0.37780784,
                                        0.61055139,
                                        0.55168478,
                                        -0.5888475,
                                        0.03445704,
                                        0.19167931,
                                        -0.3946827,
                                        -0.30889652,
                                        -0.37144644,
                                        0.31708236,
                                        0.13104209,
                                        -0.18736027,
                                        -0.22276389,
                                        0.44193283,
                                        -0.16477929,
                                        -0.11893477};

        std::vector<float> lho_data_gold{0.03445704,
                                         0.19167931,
                                         -0.3946827,
                                         -0.30889652,
                                         -0.22276389,
                                         0.44193283,
                                         -0.16477929,
                                         -0.11893477};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(lho_data, lho_data_gold));
    }

    {
        migraphx::program p;
        auto* mm      = p.get_main_module();
        auto seq_orig = mm->add_literal(migraphx::literal{in_shape, input});
        auto ih       = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto w        = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r        = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias     = mm->add_literal(migraphx::literal{b_shape, bias_data});
        migraphx::shape pad_seq_s{migraphx::shape::float_type, {batch_size, 2, input_size}};
        std::vector<float> pad_data(pad_seq_s.elements(), 0.0f);
        auto seq_p = mm->add_literal(migraphx::literal{pad_seq_s, pad_data});
        auto seq = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), seq_orig, seq_p);
        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data(batch_size, static_cast<int32_t>(seq_len));
        auto sql = mm->add_literal(seq_len_s, len_data);

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func", {}},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih);
        auto last_out = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        last_out =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_out);
        mm->add_return({out_hs, last_out});
        p.compile(migraphx::make_target("ref"));

        auto outputs         = p.eval({});
        const auto& arg_hs          = outputs.front();
        const auto& arg_last_output = outputs.back();
        std::vector<float> last_output_data;
        std::vector<float> hs_data;
        arg_hs.visit([&](auto out) { hs_data.assign(out.begin(), out.end()); });
        arg_last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> hs_data_gold{
            0.37780784,  0.61055139,  0.55168478,  -0.5888475, 0.03445704, 0.19167931,  -0.3946827,
            -0.30889652, 0.0,         0.0,         0.0,        0.0,        0.0,         0.0,
            0.0,         0.0,         -0.37144644, 0.31708236, 0.13104209, -0.18736027, -0.22276389,
            0.44193283,  -0.16477929, -0.11893477, 0.0,        0.0,        0.0,         0.0,
            0.0,         0.0,         0.0,         0.0};

        std::vector<float> last_output_data_gold{0.03445704,
                                                 0.19167931,
                                                 -0.3946827,
                                                 -0.30889652,
                                                 -0.22276389,
                                                 0.44193283,
                                                 -0.16477929,
                                                 -0.11893477};
        EXPECT(migraphx::verify::verify_rms_range(last_output_data, last_output_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data{2, 1};
        auto sql = mm->add_literal(seq_len_s, len_data);

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func", {}},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih);
        auto last_out = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        last_out =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_out);
        mm->add_return({out_hs, last_out});
        p.compile(migraphx::make_target("ref"));

        auto outputs         = p.eval({});
        const auto& arg_hs          = outputs.front();
        const auto& arg_last_output = outputs.back();
        std::vector<float> last_output_data;
        std::vector<float> hs_data;
        arg_hs.visit([&](auto out) { hs_data.assign(out.begin(), out.end()); });
        arg_last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> hs_data_gold{0.377808,
                                        0.610551,
                                        0.551685,
                                        -0.588848,
                                        0.034457,
                                        0.191679,
                                        -0.394683,
                                        -0.308897,
                                        -0.371446,
                                        0.317082,
                                        0.131042,
                                        -0.18736,
                                        0,
                                        0,
                                        0,
                                        0};
        std::vector<float> last_output_data_gold{
            0.034457, 0.191679, -0.394683, -0.308897, -0.371446, 0.317082, 0.131042, -0.18736};
        EXPECT(migraphx::verify::verify_rms_range(last_output_data, last_output_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // seq_len = 1
    {
        seq_len = 1;
        std::vector<float> input_1(seq_len * batch_size * input_size, 0);
        input_1[0] = input_1[1] = 1.0;
        migraphx::shape in_shape_1{migraphx::shape::float_type, {batch_size, seq_len, input_size}};

        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape_1, input_1});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto out_hs = mm->add_instruction(
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
            und,
            ih);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{0.37780784,
                                        0.61055139,
                                        0.55168478,
                                        -0.5888475,
                                        -0.37144644,
                                        0.31708236,
                                        0.13104209,
                                        -0.18736027};
        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(rnn_reverse)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 2;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    std::vector<float> w_data{-0.0296,
                              -0.1341,
                              0.1761,
                              -0.2325,
                              -0.0717,
                              0.1852,
                              0.2720,
                              0.1471,
                              -0.1097,
                              0.3363,
                              -0.0587,
                              -0.2302};
    std::vector<float> r_data{0.2528,
                              -0.2333,
                              0.3973,
                              0.1593,
                              -0.0388,
                              0.1702,
                              0.3829,
                              -0.0712,
                              -0.1668,
                              0.3074,
                              -0.2854,
                              0.4049,
                              -0.3737,
                              -0.1051,
                              0.4482,
                              -0.2841};
    std::vector<float> bias_data{-0.3188, 0.1341, -0.4446, 0.1389, 0.3117, 0.3664, 0.2352, 0.2552};
    std::vector<float> input(seq_len * batch_size * input_size, 0);
    input[0] = input[1] = 1.0;
    std::vector<float> ih_data(num_dirct * batch_size * hidden_size, 0);
    float clip = 0.0f;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    // concatenation of hidden states as program output
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func", {}},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.29385301,
                                        0.16796815,
                                        0.51075965,
                                        0.40258689,
                                        -0.13818839,
                                        0.44124447,
                                        0.14365635,
                                        0.14803654,
                                        -0.0070999,
                                        0.46251031,
                                        -0.20639211,
                                        0.37488942,
                                        -0.0070999,
                                        0.46251031,
                                        -0.20639211,
                                        0.37488942};
        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // rnn last output as program output
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func", {}},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);

        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        p.compile(migraphx::make_target("ref"));

        auto last_output = p.eval({}).back();
        std::vector<float> last_output_data;
        last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> last_output_data_gold{-0.29385301,
                                                 0.16796815,
                                                 0.51075965,
                                                 0.40258689,
                                                 -0.13818839,
                                                 0.44124447,
                                                 0.14365635,
                                                 0.14803654};
        EXPECT(migraphx::verify::verify_rms_range(last_output_data, last_output_data_gold));
    }

    // rnn hidden states and last hidden state output as program outputs
    {
        migraphx::program p;
        auto* mm      = p.get_main_module();
        auto seq_orig = mm->add_literal(migraphx::literal{in_shape, input});
        auto ih       = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto w        = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r        = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias     = mm->add_literal(migraphx::literal{b_shape, bias_data});
        migraphx::shape pad_seq_s{migraphx::shape::float_type, {2, batch_size, input_size}};
        std::vector<float> pad_data(pad_seq_s.elements(), 0.0f);
        auto seq_p = mm->add_literal(migraphx::literal{pad_seq_s, pad_data});
        auto seq = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), seq_orig, seq_p);
        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data(batch_size, static_cast<int32_t>(seq_len));
        auto sql = mm->add_literal(seq_len_s, len_data);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func", {}},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih);

        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        mm->add_return({out_hs, lho});
        p.compile(migraphx::make_target("ref"));

        auto outputs = p.eval({});
        std::vector<float> hs_data;
        std::vector<float> last_output_data;
        const auto& arg_hs = outputs.front();
        arg_hs.visit([&](auto out) { hs_data.assign(out.begin(), out.end()); });

        const auto& arg_lho = outputs.back();
        arg_lho.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> hs_data_gold{
            -0.29385301, 0.16796815, 0.51075965, 0.40258689,  -0.13818839, 0.44124447, 0.14365635,
            0.14803654,  -0.0070999, 0.46251031, -0.20639211, 0.37488942,  -0.0070999, 0.46251031,
            -0.20639211, 0.37488942, 0.0,        0.0,         0.0,         0.0,        0.0,
            0.0,         0.0,        0.0,        0.0,         0.0,         0.0,        0.0,
            0.0,         0.0,        0.0,        0.0};

        std::vector<float> last_output_data_gold{-0.29385301,
                                                 0.16796815,
                                                 0.51075965,
                                                 0.40258689,
                                                 -0.13818839,
                                                 0.44124447,
                                                 0.14365635,
                                                 0.14803654};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(last_output_data, last_output_data_gold));
    }

    // rnn hidden states and last hidden state output as program outputs
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data{2, 1};
        auto sql = mm->add_literal(seq_len_s, len_data);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func", {}},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih);

        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        mm->add_return({out_hs, lho});
        p.compile(migraphx::make_target("ref"));

        auto outputs = p.eval({});
        std::vector<float> hs_data;
        std::vector<float> last_output_data;
        const auto& arg_hs = outputs.front();
        arg_hs.visit([&](auto out) { hs_data.assign(out.begin(), out.end()); });

        const auto& arg_lho = outputs.back();
        arg_lho.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });
        std::vector<float> hs_data_gold{-0.293853,
                                        0.167968,
                                        0.51076,
                                        0.402587,
                                        -0.0070999,
                                        0.46251,
                                        -0.206392,
                                        0.374889,
                                        -0.0070999,
                                        0.46251,
                                        -0.206392,
                                        0.374889,
                                        0,
                                        0,
                                        0,
                                        0};
        std::vector<float> last_output_data_gold{
            -0.293853, 0.167968, 0.51076, 0.402587, -0.0070999, 0.46251, -0.206392, 0.374889};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(last_output_data, last_output_data_gold));
    }
}

TEST_CASE(rnn_reverse_layout)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 2;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    std::vector<float> w_data{-0.0296,
                              -0.1341,
                              0.1761,
                              -0.2325,
                              -0.0717,
                              0.1852,
                              0.2720,
                              0.1471,
                              -0.1097,
                              0.3363,
                              -0.0587,
                              -0.2302};
    std::vector<float> r_data{0.2528,
                              -0.2333,
                              0.3973,
                              0.1593,
                              -0.0388,
                              0.1702,
                              0.3829,
                              -0.0712,
                              -0.1668,
                              0.3074,
                              -0.2854,
                              0.4049,
                              -0.3737,
                              -0.1051,
                              0.4482,
                              -0.2841};
    std::vector<float> bias_data{-0.3188, 0.1341, -0.4446, 0.1389, 0.3117, 0.3664, 0.2352, 0.2552};
    std::vector<float> input(seq_len * batch_size * input_size, 0);
    input[0] = input[1] = 1.0;
    std::vector<float> ih_data(num_dirct * batch_size * hidden_size, 0);
    float clip = 0.0f;
    migraphx::shape in_shape{migraphx::shape::float_type, {batch_size, seq_len, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {batch_size, num_dirct, hidden_size}};
    // concatenation of hidden states as program output
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func", {}},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}), hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.29385301,
                                        0.16796815,
                                        0.51075965,
                                        0.40258689,
                                        -0.0070999,
                                        0.46251031,
                                        -0.20639211,
                                        0.37488942,
                                        -0.13818839,
                                        0.44124447,
                                        0.14365635,
                                        0.14803654,
                                        -0.0070999,
                                        0.46251031,
                                        -0.20639211,
                                        0.37488942};
        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // rnn last output as program output
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func", {}},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);

        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        lho = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), lho);
        p.compile(migraphx::make_target("ref"));

        auto last_output = p.eval({}).back();
        std::vector<float> last_output_data;
        last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> last_output_data_gold{-0.29385301,
                                                 0.16796815,
                                                 0.51075965,
                                                 0.40258689,
                                                 -0.13818839,
                                                 0.44124447,
                                                 0.14365635,
                                                 0.14803654};
        EXPECT(migraphx::verify::verify_rms_range(last_output_data, last_output_data_gold));
    }

    // rnn hidden states and last hidden state output as program outputs
    {
        migraphx::program p;
        auto* mm      = p.get_main_module();
        auto seq_orig = mm->add_literal(migraphx::literal{in_shape, input});
        auto ih       = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto w        = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r        = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias     = mm->add_literal(migraphx::literal{b_shape, bias_data});
        migraphx::shape pad_seq_s{migraphx::shape::float_type, {batch_size, 2, input_size}};
        std::vector<float> pad_data(pad_seq_s.elements(), 0.0f);
        auto seq_p = mm->add_literal(migraphx::literal{pad_seq_s, pad_data});
        auto seq = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), seq_orig, seq_p);
        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data(batch_size, static_cast<int32_t>(seq_len));
        auto sql = mm->add_literal(seq_len_s, len_data);

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func", {}},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih);

        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        lho    = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), lho);
        mm->add_return({out_hs, lho});
        p.compile(migraphx::make_target("ref"));

        auto outputs = p.eval({});
        std::vector<float> hs_data;
        std::vector<float> last_output_data;
        const auto& arg_hs = outputs.front();
        arg_hs.visit([&](auto out) { hs_data.assign(out.begin(), out.end()); });

        const auto& arg_lho = outputs.back();
        arg_lho.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> hs_data_gold{
            -0.29385301, 0.16796815,  0.51075965,  0.40258689, -0.0070999, 0.46251031, -0.20639211,
            0.37488942,  0.0,         0.0,         0.0,        0.0,        0.0,        0.0,
            0.0,         0.0,         -0.13818839, 0.44124447, 0.14365635, 0.14803654, -0.0070999,
            0.46251031,  -0.20639211, 0.37488942,  0.0,        0.0,        0.0,        0.0,
            0.0,         0.0,         0.0,         0.0};

        std::vector<float> last_output_data_gold{-0.29385301,
                                                 0.16796815,
                                                 0.51075965,
                                                 0.40258689,
                                                 -0.13818839,
                                                 0.44124447,
                                                 0.14365635,
                                                 0.14803654};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(last_output_data, last_output_data_gold));
    }

    // rnn hidden states and last hidden state output as program outputs
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data{2, 1};
        auto sql = mm->add_literal(seq_len_s, len_data);

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func", {}},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih);

        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        lho    = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), lho);
        mm->add_return({out_hs, lho});
        p.compile(migraphx::make_target("ref"));

        auto outputs = p.eval({});
        std::vector<float> hs_data;
        std::vector<float> last_output_data;
        const auto& arg_hs = outputs.front();
        arg_hs.visit([&](auto out) { hs_data.assign(out.begin(), out.end()); });

        const auto& arg_lho = outputs.back();
        arg_lho.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });
        std::vector<float> hs_data_gold{-0.293853,
                                        0.167968,
                                        0.51076,
                                        0.402587,
                                        -0.0070999,
                                        0.46251,
                                        -0.206392,
                                        0.374889,
                                        -0.0070999,
                                        0.46251,
                                        -0.206392,
                                        0.374889,
                                        0,
                                        0,
                                        0,
                                        0};
        std::vector<float> last_output_data_gold{
            -0.293853, 0.167968, 0.51076, 0.402587, -0.0070999, 0.46251, -0.206392, 0.374889};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(last_output_data, last_output_data_gold));
    }
}

TEST_CASE(rnn_bidirectional)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 2;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 2;
    std::vector<float> w_data{0.4691,  0.3185,  -0.2227, 0.4423,  -0.0609, -0.2803,
                              0.1744,  0.3146,  0.4049,  -0.3973, -0.0890, -0.1636,
                              -0.0296, -0.1341, 0.1761,  -0.2325, -0.0717, 0.1852,
                              0.2720,  0.1471,  -0.1097, 0.3363,  -0.0587, -0.2302};

    std::vector<float> r_data{-0.0456, 0.1061,  0.1574,  -0.4928, -0.4300, -0.1909, -0.0225,
                              -0.2668, 0.1840,  -0.4453, -0.4896, 0.1302,  -0.0929, 0.3545,
                              -0.4981, 0.0616,  0.2528,  -0.2333, 0.3973,  0.1593,  -0.0388,
                              0.1702,  0.3829,  -0.0712, -0.1668, 0.3074,  -0.2854, 0.4049,
                              -0.3737, -0.1051, 0.4482,  -0.2841};

    std::vector<float> bias_data{-0.4938,
                                 0.4355,
                                 -0.3186,
                                 0.2094,
                                 0.1037,
                                 -0.1071,
                                 0.4504,
                                 -0.3990,
                                 -0.3188,
                                 0.1341,
                                 -0.4446,
                                 0.1389,
                                 0.3117,
                                 0.3664,
                                 0.2352,
                                 0.2552};

    std::vector<float> input(seq_len * batch_size * input_size, 0);
    input[0] = input[1] = 1.0;
    std::vector<float> ih_data(num_dirct * batch_size * hidden_size, 0);

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
    float clip = 0.0f;
    // concatenation of hidden state and last hs output for program outputs
    {

        migraphx::program p;
        auto* mm    = p.get_main_module();
        auto seq    = mm->add_literal(migraphx::literal{in_shape, input});
        auto ih     = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto w      = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r      = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias   = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und    = mm->add_instruction(migraphx::make_op("undefined"));
        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func", {}},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        mm->add_return({out_hs, lho});
        p.compile(migraphx::make_target("ref"));

        auto outputs = p.eval({});
        const auto& arg_hs  = outputs.front();
        const auto& arg_lho = outputs.back();

        std::vector<float> hs_data;
        arg_hs.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });
        std::vector<float> last_output_data;
        arg_lho.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> hs_data_gold{
            0.37780784,  0.61055139,  0.55168478,  -0.5888475, -0.37144644, 0.31708236,
            0.13104209,  -0.18736027, -0.29385301, 0.16796815, 0.51075965,  0.40258689,
            -0.13818839, 0.44124447,  0.14365635,  0.14803654, 0.03445704,  0.19167931,
            -0.3946827,  -0.30889652, -0.22276389, 0.44193283, -0.16477929, -0.11893477,
            -0.0070999,  0.46251031,  -0.20639211, 0.37488942, -0.0070999,  0.46251031,
            -0.20639211, 0.37488942};

        std::vector<float> last_output_data_gold{0.03445704,
                                                 0.19167931,
                                                 -0.3946827,
                                                 -0.30889652,
                                                 -0.22276389,
                                                 0.44193283,
                                                 -0.16477929,
                                                 -0.11893477,
                                                 -0.29385301,
                                                 0.16796815,
                                                 0.51075965,
                                                 0.40258689,
                                                 -0.13818839,
                                                 0.44124447,
                                                 0.14365635,
                                                 0.14803654};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(last_output_data, last_output_data_gold));
    }

    // last rnn output for program output
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data{1, 2};
        auto sql = mm->add_literal(seq_len_s, len_data);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        mm->add_return({out_hs, lho});
        p.compile(migraphx::make_target("ref"));

        auto outputs = p.eval({});
        const auto& arg_hs  = outputs.front();
        const auto& arg_lho = outputs.back();

        std::vector<float> hs_data;
        std::vector<float> last_output_data;
        arg_hs.visit([&](auto out) { hs_data.assign(out.begin(), out.end()); });
        arg_lho.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> hs_data_gold{
            0.377808,  0.610551, 0.551685, -0.588848, -0.371446,  0.317082, 0.131042,  -0.18736,
            -0.169158, 0.193817, 0.206679, 0.586097,  -0.138188,  0.441244, 0.143656,  0.148037,
            0,         0,        0,        0,         -0.222764,  0.441933, -0.164779, -0.118935,
            0,         0,        0,        0,         -0.0070999, 0.46251,  -0.206392, 0.374889};
        std::vector<float> last_output_data_gold{0.377808,
                                                 0.610551,
                                                 0.551685,
                                                 -0.588848,
                                                 -0.222764,
                                                 0.441933,
                                                 -0.164779,
                                                 -0.118935,
                                                 -0.169158,
                                                 0.193817,
                                                 0.206679,
                                                 0.586097,
                                                 -0.138188,
                                                 0.441244,
                                                 0.143656,
                                                 0.148037};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(last_output_data, last_output_data_gold));
    }

    // 4 args
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias);

        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        p.compile(migraphx::make_target("ref"));

        auto last_output = p.eval({}).back();
        std::vector<float> last_output_data;
        last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> last_output_data_gold{0.03445704,
                                                 0.19167931,
                                                 -0.3946827,
                                                 -0.30889652,
                                                 -0.22276389,
                                                 0.44193283,
                                                 -0.16477929,
                                                 -0.11893477,
                                                 -0.29385301,
                                                 0.16796815,
                                                 0.51075965,
                                                 0.40258689,
                                                 -0.13818839,
                                                 0.44124447,
                                                 0.14365635,
                                                 0.14803654};

        EXPECT(migraphx::verify::verify_rms_range(last_output_data, last_output_data_gold));
    }

    // 3 args
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input});
        auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});

        mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            seq,
            w,
            r);
        p.compile(migraphx::make_target("ref"));

        auto last_output = p.eval({}).back();
        std::vector<float> last_output_data;
        last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> last_output_data_gold{
            0.6570473,   0.36392266,  0.45342238,  -0.45127486, 0., 0., 0., 0.,
            -0.16225325, -0.29515147, 0.39617197,  0.27068236,  0., 0., 0., 0.,
            0.2935145,   -0.23719997, -0.31123261, -0.18357255, 0., 0., 0., 0.,
            0.,          0.,          0.,          0.,          0., 0., 0., 0.};

        EXPECT(migraphx::verify::verify_rms_range(last_output_data, last_output_data_gold));
    }

    // concatenation of hidden state for program output
    {
        seq_len = 1;
        std::vector<float> input_1(seq_len * batch_size * input_size, 0);
        input_1[0] = input_1[1] = 1.0;
        migraphx::shape in_shape_1{migraphx::shape::float_type, {seq_len, batch_size, input_size}};

        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape_1, input_1});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func", {}},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{0.37780784,
                                        0.61055139,
                                        0.55168478,
                                        -0.5888475,
                                        -0.37144644,
                                        0.31708236,
                                        0.13104209,
                                        -0.18736027,
                                        -0.16915828,
                                        0.1938169,
                                        0.20667936,
                                        0.58609703,
                                        -0.0070999,
                                        0.46251031,
                                        -0.20639211,
                                        0.37488942};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(rnn_bidirectional_layout)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 2;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 2;
    std::vector<float> w_data{0.4691,  0.3185,  -0.2227, 0.4423,  -0.0609, -0.2803,
                              0.1744,  0.3146,  0.4049,  -0.3973, -0.0890, -0.1636,
                              -0.0296, -0.1341, 0.1761,  -0.2325, -0.0717, 0.1852,
                              0.2720,  0.1471,  -0.1097, 0.3363,  -0.0587, -0.2302};

    std::vector<float> r_data{-0.0456, 0.1061,  0.1574,  -0.4928, -0.4300, -0.1909, -0.0225,
                              -0.2668, 0.1840,  -0.4453, -0.4896, 0.1302,  -0.0929, 0.3545,
                              -0.4981, 0.0616,  0.2528,  -0.2333, 0.3973,  0.1593,  -0.0388,
                              0.1702,  0.3829,  -0.0712, -0.1668, 0.3074,  -0.2854, 0.4049,
                              -0.3737, -0.1051, 0.4482,  -0.2841};

    std::vector<float> bias_data{-0.4938,
                                 0.4355,
                                 -0.3186,
                                 0.2094,
                                 0.1037,
                                 -0.1071,
                                 0.4504,
                                 -0.3990,
                                 -0.3188,
                                 0.1341,
                                 -0.4446,
                                 0.1389,
                                 0.3117,
                                 0.3664,
                                 0.2352,
                                 0.2552};

    std::vector<float> input(seq_len * batch_size * input_size, 0);
    input[0] = input[1] = 1.0;
    std::vector<float> ih_data(num_dirct * batch_size * hidden_size, 0);

    migraphx::shape in_shape{migraphx::shape::float_type, {batch_size, seq_len, input_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {batch_size, num_dirct, hidden_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
    float clip = 0.0f;
    // concatenation of hidden state and last hs output for program outputs
    {

        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func", {}},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        lho    = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), lho);
        mm->add_return({out_hs, lho});
        p.compile(migraphx::make_target("ref"));

        auto outputs = p.eval({});
        const auto& arg_hs  = outputs.front();
        const auto& arg_lho = outputs.back();

        std::vector<float> hs_data;
        arg_hs.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });
        std::vector<float> last_output_data;
        arg_lho.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> hs_data_gold{
            0.37780784,  0.61055139,  0.55168478,  -0.5888475,  -0.29385301, 0.16796815,
            0.51075965,  0.40258689,  0.03445704,  0.19167931,  -0.3946827,  -0.30889652,
            -0.0070999,  0.46251031,  -0.20639211, 0.37488942,  -0.37144644, 0.31708236,
            0.13104209,  -0.18736027, -0.13818839, 0.44124447,  0.14365635,  0.14803654,
            -0.22276389, 0.44193283,  -0.16477929, -0.11893477, -0.0070999,  0.46251031,
            -0.20639211, 0.37488942};

        std::vector<float> last_output_data_gold{0.03445704,
                                                 0.19167931,
                                                 -0.3946827,
                                                 -0.30889652,
                                                 -0.29385301,
                                                 0.16796815,
                                                 0.51075965,
                                                 0.40258689,
                                                 -0.22276389,
                                                 0.44193283,
                                                 -0.16477929,
                                                 -0.11893477,
                                                 -0.13818839,
                                                 0.44124447,
                                                 0.14365635,
                                                 0.14803654};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(last_output_data, last_output_data_gold));
    }

    // last rnn output for program output
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data{1, 2};
        auto sql = mm->add_literal(seq_len_s, len_data);

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        lho    = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), lho);
        mm->add_return({out_hs, lho});
        p.compile(migraphx::make_target("ref"));

        auto outputs = p.eval({});
        const auto& arg_hs  = outputs.front();
        const auto& arg_lho = outputs.back();

        std::vector<float> hs_data;
        std::vector<float> last_output_data;
        arg_hs.visit([&](auto out) { hs_data.assign(out.begin(), out.end()); });
        arg_lho.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> hs_data_gold{
            0.377808,  0.610551, 0.551685,  -0.588848, -0.169158,  0.193817, 0.206679,  0.586097,
            0,         0,        0,         0,         0,          0,        0,         0,
            -0.371446, 0.317082, 0.131042,  -0.18736,  -0.138188,  0.441244, 0.143656,  0.148037,
            -0.222764, 0.441933, -0.164779, -0.118935, -0.0070999, 0.46251,  -0.206392, 0.374889};

        std::vector<float> last_output_data_gold{0.377808,
                                                 0.610551,
                                                 0.551685,
                                                 -0.588848,
                                                 -0.169158,
                                                 0.193817,
                                                 0.206679,
                                                 0.586097,
                                                 -0.222764,
                                                 0.441933,
                                                 -0.164779,
                                                 -0.118935,
                                                 -0.138188,
                                                 0.441244,
                                                 0.143656,
                                                 0.148037};
        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(last_output_data, last_output_data_gold));
    }

    // 4 args
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        lho = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), lho);
        p.compile(migraphx::make_target("ref"));

        auto last_output = p.eval({}).back();
        std::vector<float> last_output_data;
        last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> last_output_data_gold{0.03445704,
                                                 0.19167931,
                                                 -0.3946827,
                                                 -0.30889652,
                                                 -0.29385301,
                                                 0.16796815,
                                                 0.51075965,
                                                 0.40258689,
                                                 -0.22276389,
                                                 0.44193283,
                                                 -0.16477929,
                                                 -0.11893477,
                                                 -0.13818839,
                                                 0.44124447,
                                                 0.14365635,
                                                 0.14803654};
        EXPECT(migraphx::verify::verify_rms_range(last_output_data, last_output_data_gold));
    }

    // 3 args
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input});
        auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            seq,
            w,
            r);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        p.compile(migraphx::make_target("ref"));

        auto last_output = p.eval({}).back();
        std::vector<float> last_output_data;
        last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> last_output_data_gold{
            0.6570473,  0.36392266, 0.45342238,  -0.45127486, -0.16225325, -0.29515147, 0.39617197,
            0.27068236, 0.2935145,  -0.23719997, -0.31123261, -0.18357255, 0.,          0.,
            0.,         0.,         0.,          0.,          0.,          0.,          0.,
            0.,         0.,         0.,          0.,          0.,          0.,          0.,
            0.,         0.,         0.,          0.};

        EXPECT(migraphx::verify::verify_rms_range(last_output_data, last_output_data_gold));
    }

    // concatenation of hidden state for program output
    {
        seq_len = 1;
        std::vector<float> input_1(seq_len * batch_size * input_size, 0);
        input_1[0] = input_1[1] = 1.0;
        migraphx::shape in_shape_1{migraphx::shape::float_type, {batch_size, seq_len, input_size}};

        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape_1, input_1});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func", {}},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{0.37780784,
                                        0.61055139,
                                        0.55168478,
                                        -0.5888475,
                                        -0.16915828,
                                        0.1938169,
                                        0.20667936,
                                        0.58609703,
                                        -0.37144644,
                                        0.31708236,
                                        0.13104209,
                                        -0.18736027,
                                        -0.0070999,
                                        0.46251031,
                                        -0.20639211,
                                        0.37488942};
        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(rnn_fp16)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 2;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    std::vector<float> w_data{0.4691,
                              0.3185,
                              -0.2227,
                              0.4423,
                              -0.0609,
                              -0.2803,
                              0.1744,
                              0.3146,
                              0.4049,
                              -0.3973,
                              -0.0890,
                              -0.1636};

    std::vector<float> r_data{-0.0456,
                              0.1061,
                              0.1574,
                              -0.4928,
                              -0.4300,
                              -0.1909,
                              -0.0225,
                              -0.2668,
                              0.1840,
                              -0.4453,
                              -0.4896,
                              0.1302,
                              -0.0929,
                              0.3545,
                              -0.4981,
                              0.0616};

    std::vector<float> bias_data{
        -0.4938, 0.4355, -0.3186, 0.2094, 0.1037, -0.1071, 0.4504, -0.3990};
    std::vector<float> ih_data(num_dirct * batch_size * hidden_size, 0);
    std::vector<float> input(seq_len * batch_size * input_size, 0);
    input[0] = input[1] = 1.0;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
    float clip = 0.0f;

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto seq = mm->add_literal(migraphx::literal{in_shape, input});
    auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
    auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});

    auto seq_half = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), seq);
    auto w_half = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), w);
    auto r_half = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), r);

    auto out_hs = mm->add_instruction(
        migraphx::make_op("rnn",
                          {{"hidden_size", hidden_size},
                           {"actv_func", {}},
                           {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                           {"clip", clip}}),
        seq_half,
        w_half,
        r_half);
    mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);

    p.compile(migraphx::make_target("ref"));

    auto last_output = p.eval({}).back();
    std::vector<float> last_output_data;
    last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

    std::vector<float> last_output_data_gold{
        0.2935145, -0.23719997, -0.31123261, -0.18357255, 0., 0., 0., 0.};
    EXPECT(migraphx::verify::verify_range_with_tolerance(
        last_output_data,
        migraphx::verify::expected{last_output_data_gold},
        migraphx::verify::tolerance{0.005}));
}

TEST_CASE(gru_forward)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, input_size}};
    std::vector<float> w_data{
        0.3485,  -0.0378, -0.1782, 0.1416,  -0.3096, -0.2212, -0.3883, 0.1983,  -0.2418,
        0.1480,  -0.3255, 0.1359,  -0.3551, -0.3605, -0.3482, -0.1424, -0.0495, -0.1640,
        -0.1979, -0.2577, -0.4097, -0.1211, -0.0412, 0.1801,  0.1721,  -0.4327, -0.0498,
        0.2628,  -0.1573, -0.1577, 0.2759,  -0.2023, -0.1185, -0.2136, 0.1294,  -0.2331,
        0.0701,  0.4316,  0.0480,  0.0247,  -0.0166, -0.2729, 0.1712,  -0.3984, -0.3905};

    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, hidden_size}};
    std::vector<float> r_data{
        0.2848,  -0.2851, -0.3466, -0.1718, -0.1492, -0.0082, 0.2452,  -0.0401, 0.3399,  0.2529,
        -0.0953, -0.0903, -0.1518, -0.1373, 0.3848,  -0.0130, -0.4339, 0.0406,  -0.1926, -0.1131,
        0.4285,  -0.0013, 0.2243,  0.2752,  0.1776,  -0.1720, 0.0822,  -0.0295, 0.1062,  -0.2721,
        -0.2736, -0.1826, 0.3541,  -0.4259, 0.2188,  0.0706,  0.3650,  0.3947,  0.2522,  0.2179,
        -0.0744, 0.2122,  -0.4346, 0.2760,  0.4076,  0.1183,  -0.1500, -0.1704, 0.3090,  -0.0706,
        -0.2442, 0.3021,  0.1680,  0.0783,  -0.3754, -0.3469, -0.2972, -0.0170, 0.4143,  0.3801,
        0.3852,  -0.1170, -0.2937, 0.2979,  -0.1357, 0.4257,  0.3884,  -0.2916, 0.1071,  0.0934,
        0.3645,  -0.4310, -0.3480, 0.0702,  -0.1558};

    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
    std::vector<float> bias_data{
        0.0560,  0.0310, -0.1669, -0.0781, 0.1793, -0.1758, 0.3173,  -0.1650, -0.3732, 0.2946,
        -0.0912, 0.3118, 0.1391,  0.2755,  0.2695, -0.1059, -0.2357, 0.3629,  -0.2534, -0.0494,
        0.0556,  0.0881, -0.2592, -0.2213, 0.2310, -0.4044, 0.1801,  0.1438,  0.3108,  -0.3607};

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    std::vector<float> input{-0.8432,
                             -0.9887,
                             1.3041,
                             -2.6430,
                             -0.3306,
                             -0.8504,
                             -0.3933,
                             0.5151,
                             -0.2951,
                             0.0093,
                             -1.1948,
                             -0.1239,
                             0.0373,
                             1.3211,
                             0.7854,
                             -0.4838,
                             -1.0536,
                             -0.2529};

    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    std::vector<float> ih_data{
        -0.0468, 0.5691, -0.0882, 0.8340, 0.1483, -0.3902, -0.5348, 0.4178, 1.0175, 0.9212};
    float clip = 0.0f;
    // concatenation of hidden states for output
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);

        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            -0.27298412, 0.42363745,  -0.09368783, 0.4823072,   -0.02183238, -0.6873896,
            0.16144305,  0.31932795,  0.6104771,   0.79759157,  -0.31791314, 0.5249062,
            0.08800987,  0.46404213,  -0.11872687, -0.26210734, 0.34448293,  -0.0176422,
            0.48523626,  0.60002893,  -0.3969709,  0.43360898,  0.35775262,  0.23280787,
            -0.52179873, -0.21944991, 0.4535257,   -0.13735442, 0.51757574,  0.50380427};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // last output for output
    {
        migraphx::program p;
        auto* mm       = p.get_main_module();
        auto seq       = mm->add_literal(migraphx::literal{in_shape, input});
        auto w         = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r         = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias      = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und       = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih        = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), concat_hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.3969709,
                                        0.43360898,
                                        0.35775262,
                                        0.23280787,
                                        -0.52179873,
                                        -0.21944991,
                                        0.4535257,
                                        -0.13735442,
                                        0.51757574,
                                        0.50380427};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // two rnn_last_hs_output operators after gru
    {
        migraphx::program p;
        auto* mm       = p.get_main_module();
        auto seq       = mm->add_literal(migraphx::literal{in_shape, input});
        auto w         = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r         = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias      = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und       = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih        = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), concat_hs);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), concat_hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.3969709,
                                        0.43360898,
                                        0.35775262,
                                        0.23280787,
                                        -0.52179873,
                                        -0.21944991,
                                        0.4535257,
                                        -0.13735442,
                                        0.51757574,
                                        0.50380427};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // last output for output, linear_before_reset = 0
    {
        migraphx::program p;
        auto* mm       = p.get_main_module();
        auto seq       = mm->add_literal(migraphx::literal{in_shape, input});
        auto w         = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r         = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias      = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und       = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih        = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"linear_before_reset", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), concat_hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.53291196,
                                        0.50160867,
                                        0.39010462,
                                        0.39292926,
                                        -0.5960838,
                                        -0.38451535,
                                        0.454239,
                                        -0.10620412,
                                        0.6014447,
                                        0.43445644};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(gru_forward_layout)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, input_size}};
    std::vector<float> w_data{
        0.3485,  -0.0378, -0.1782, 0.1416,  -0.3096, -0.2212, -0.3883, 0.1983,  -0.2418,
        0.1480,  -0.3255, 0.1359,  -0.3551, -0.3605, -0.3482, -0.1424, -0.0495, -0.1640,
        -0.1979, -0.2577, -0.4097, -0.1211, -0.0412, 0.1801,  0.1721,  -0.4327, -0.0498,
        0.2628,  -0.1573, -0.1577, 0.2759,  -0.2023, -0.1185, -0.2136, 0.1294,  -0.2331,
        0.0701,  0.4316,  0.0480,  0.0247,  -0.0166, -0.2729, 0.1712,  -0.3984, -0.3905};

    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, hidden_size}};
    std::vector<float> r_data{
        0.2848,  -0.2851, -0.3466, -0.1718, -0.1492, -0.0082, 0.2452,  -0.0401, 0.3399,  0.2529,
        -0.0953, -0.0903, -0.1518, -0.1373, 0.3848,  -0.0130, -0.4339, 0.0406,  -0.1926, -0.1131,
        0.4285,  -0.0013, 0.2243,  0.2752,  0.1776,  -0.1720, 0.0822,  -0.0295, 0.1062,  -0.2721,
        -0.2736, -0.1826, 0.3541,  -0.4259, 0.2188,  0.0706,  0.3650,  0.3947,  0.2522,  0.2179,
        -0.0744, 0.2122,  -0.4346, 0.2760,  0.4076,  0.1183,  -0.1500, -0.1704, 0.3090,  -0.0706,
        -0.2442, 0.3021,  0.1680,  0.0783,  -0.3754, -0.3469, -0.2972, -0.0170, 0.4143,  0.3801,
        0.3852,  -0.1170, -0.2937, 0.2979,  -0.1357, 0.4257,  0.3884,  -0.2916, 0.1071,  0.0934,
        0.3645,  -0.4310, -0.3480, 0.0702,  -0.1558};

    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
    std::vector<float> bias_data{
        0.0560,  0.0310, -0.1669, -0.0781, 0.1793, -0.1758, 0.3173,  -0.1650, -0.3732, 0.2946,
        -0.0912, 0.3118, 0.1391,  0.2755,  0.2695, -0.1059, -0.2357, 0.3629,  -0.2534, -0.0494,
        0.0556,  0.0881, -0.2592, -0.2213, 0.2310, -0.4044, 0.1801,  0.1438,  0.3108,  -0.3607};

    migraphx::shape in_shape{migraphx::shape::float_type, {batch_size, seq_len, input_size}};
    std::vector<float> input{-0.8432,
                             -0.9887,
                             1.3041,
                             -0.3933,
                             0.5151,
                             -0.2951,
                             0.0373,
                             1.3211,
                             0.7854,
                             -2.6430,
                             -0.3306,
                             -0.8504,
                             0.0093,
                             -1.1948,
                             -0.1239,
                             -0.4838,
                             -1.0536,
                             -0.2529};

    migraphx::shape ih_shape{migraphx::shape::float_type, {batch_size, num_dirct, hidden_size}};
    std::vector<float> ih_data{
        -0.0468, 0.5691, -0.0882, 0.8340, 0.1483, -0.3902, -0.5348, 0.4178, 1.0175, 0.9212};
    float clip = 0.0f;
    // concatenation of hidden states for output
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}), hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            -0.27298412, 0.42363745,  -0.09368783, 0.4823072,   -0.02183238, -0.31791314,
            0.5249062,   0.08800987,  0.46404213,  -0.11872687, -0.3969709,  0.43360898,
            0.35775262,  0.23280787,  -0.52179873, -0.6873896,  0.16144305,  0.31932795,
            0.6104771,   0.79759157,  -0.26210734, 0.34448293,  -0.0176422,  0.48523626,
            0.60002893,  -0.21944991, 0.4535257,   -0.13735442, 0.51757574,  0.50380427};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // last output for output
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);
        auto concat_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        auto last_output = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), concat_hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_output);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.3969709,
                                        0.43360898,
                                        0.35775262,
                                        0.23280787,
                                        -0.52179873,
                                        -0.21944991,
                                        0.4535257,
                                        -0.13735442,
                                        0.51757574,
                                        0.50380427};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // two rnn_last_hs_output operators after gru
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto concat_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        auto last_output = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), concat_hs);
        last_output      = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), concat_hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_output);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.3969709,
                                        0.43360898,
                                        0.35775262,
                                        0.23280787,
                                        -0.52179873,
                                        -0.21944991,
                                        0.4535257,
                                        -0.13735442,
                                        0.51757574,
                                        0.50380427};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // last output for output, linear_before_reset = 0
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto concat_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"linear_before_reset", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        auto last_output = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), concat_hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_output);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.53291196,
                                        0.50160867,
                                        0.39010462,
                                        0.39292926,
                                        -0.5960838,
                                        -0.38451535,
                                        0.454239,
                                        -0.10620412,
                                        0.6014447,
                                        0.43445644};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(gru_forward_args)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, input_size}};
    std::vector<float> w_data{
        0.3485,  -0.0378, -0.1782, 0.1416,  -0.3096, -0.2212, -0.3883, 0.1983,  -0.2418,
        0.1480,  -0.3255, 0.1359,  -0.3551, -0.3605, -0.3482, -0.1424, -0.0495, -0.1640,
        -0.1979, -0.2577, -0.4097, -0.1211, -0.0412, 0.1801,  0.1721,  -0.4327, -0.0498,
        0.2628,  -0.1573, -0.1577, 0.2759,  -0.2023, -0.1185, -0.2136, 0.1294,  -0.2331,
        0.0701,  0.4316,  0.0480,  0.0247,  -0.0166, -0.2729, 0.1712,  -0.3984, -0.3905};

    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, hidden_size}};
    std::vector<float> r_data{
        0.2848,  -0.2851, -0.3466, -0.1718, -0.1492, -0.0082, 0.2452,  -0.0401, 0.3399,  0.2529,
        -0.0953, -0.0903, -0.1518, -0.1373, 0.3848,  -0.0130, -0.4339, 0.0406,  -0.1926, -0.1131,
        0.4285,  -0.0013, 0.2243,  0.2752,  0.1776,  -0.1720, 0.0822,  -0.0295, 0.1062,  -0.2721,
        -0.2736, -0.1826, 0.3541,  -0.4259, 0.2188,  0.0706,  0.3650,  0.3947,  0.2522,  0.2179,
        -0.0744, 0.2122,  -0.4346, 0.2760,  0.4076,  0.1183,  -0.1500, -0.1704, 0.3090,  -0.0706,
        -0.2442, 0.3021,  0.1680,  0.0783,  -0.3754, -0.3469, -0.2972, -0.0170, 0.4143,  0.3801,
        0.3852,  -0.1170, -0.2937, 0.2979,  -0.1357, 0.4257,  0.3884,  -0.2916, 0.1071,  0.0934,
        0.3645,  -0.4310, -0.3480, 0.0702,  -0.1558};

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    std::vector<float> input{-0.8432,
                             -0.9887,
                             1.3041,
                             -2.6430,
                             -0.3306,
                             -0.8504,
                             -0.3933,
                             0.5151,
                             -0.2951,
                             0.0093,
                             -1.1948,
                             -0.1239,
                             0.0373,
                             1.3211,
                             0.7854,
                             -0.4838,
                             -1.0536,
                             -0.2529};

    float clip = 0.0f;

    // 3 args
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input});
        auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});
        mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r);

        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.114674, -0.129581,  -0.218156,  -0.140788,  -0.114242,
                                        -0.346569, 0.321367,   -0.0838253, 0.102097,   0.00232137,
                                        -0.149055, 0.0590743,  -0.0533094, -0.0446122, -0.112588,
                                        0.0153261, 0.168883,   -0.326836,  0.0843562,  0.160872,
                                        -0.232523, 0.00214573, 0.231693,   -0.160475,  -0.518952,
                                        0.0467166, 0.12327,    -0.374162,  0.137778,   0.251976};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // 4 args (bias is used)
    {
        std::vector<float> bias_data{
            0.0560,  0.0310, -0.1669, -0.0781, 0.1793, -0.1758, 0.3173,  -0.1650, -0.3732, 0.2946,
            -0.0912, 0.3118, 0.1391,  0.2755,  0.2695, -0.1059, -0.2357, 0.3629,  -0.2534, -0.0494,
            0.0556,  0.0881, -0.2592, -0.2213, 0.2310, -0.4044, 0.1801,  0.1438,  0.3108,  -0.3607};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias);

        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.273619, 0.0931375, -0.104717,  0.0203752, -0.0797887,
                                        -0.493948, 0.472118,  -0.0336318, 0.332706,  0.0182268,
                                        -0.341684, 0.38063,   0.0589275,  0.2644,    -0.115737,
                                        -0.152324, 0.442277,  -0.201626,  0.408909,  0.12905,
                                        -0.416866, 0.377186,  0.32922,    0.162214,  -0.519973,
                                        -0.140072, 0.465076,  -0.229563,  0.500164,  0.195166};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // 4 args (ih is used)
    {
        std::vector<float> ih_data{
            -0.0468, 0.5691, -0.0882, 0.8340, 0.1483, -0.3902, -0.5348, 0.4178, 1.0175, 0.9212};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input});
        auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto ih  = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto und = mm->add_instruction(migraphx::make_op("undefined"));
        mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            und,
            und,
            ih);

        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.0801064, 0.27025,   -0.20704,   0.333579,   -0.0452438,
                                        -0.56265,   0.061061,  0.262172,   0.405193,   0.775226,
                                        -0.100683,  0.258729,  -0.0187297, 0.215815,   -0.108936,
                                        -0.0941018, 0.129665,  -0.159421,  0.190636,   0.597412,
                                        -0.197,     0.0885705, 0.269396,   -0.0414511, -0.515137,
                                        -0.03075,   0.158326,  -0.296488,  0.177983,   0.519498};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(gru_forward_actv_funcs)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, input_size}};
    std::vector<float> w_data{
        0.3485,  -0.0378, -0.1782, 0.1416,  -0.3096, -0.2212, -0.3883, 0.1983,  -0.2418,
        0.1480,  -0.3255, 0.1359,  -0.3551, -0.3605, -0.3482, -0.1424, -0.0495, -0.1640,
        -0.1979, -0.2577, -0.4097, -0.1211, -0.0412, 0.1801,  0.1721,  -0.4327, -0.0498,
        0.2628,  -0.1573, -0.1577, 0.2759,  -0.2023, -0.1185, -0.2136, 0.1294,  -0.2331,
        0.0701,  0.4316,  0.0480,  0.0247,  -0.0166, -0.2729, 0.1712,  -0.3984, -0.3905};

    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, hidden_size}};
    std::vector<float> r_data{
        0.2848,  -0.2851, -0.3466, -0.1718, -0.1492, -0.0082, 0.2452,  -0.0401, 0.3399,  0.2529,
        -0.0953, -0.0903, -0.1518, -0.1373, 0.3848,  -0.0130, -0.4339, 0.0406,  -0.1926, -0.1131,
        0.4285,  -0.0013, 0.2243,  0.2752,  0.1776,  -0.1720, 0.0822,  -0.0295, 0.1062,  -0.2721,
        -0.2736, -0.1826, 0.3541,  -0.4259, 0.2188,  0.0706,  0.3650,  0.3947,  0.2522,  0.2179,
        -0.0744, 0.2122,  -0.4346, 0.2760,  0.4076,  0.1183,  -0.1500, -0.1704, 0.3090,  -0.0706,
        -0.2442, 0.3021,  0.1680,  0.0783,  -0.3754, -0.3469, -0.2972, -0.0170, 0.4143,  0.3801,
        0.3852,  -0.1170, -0.2937, 0.2979,  -0.1357, 0.4257,  0.3884,  -0.2916, 0.1071,  0.0934,
        0.3645,  -0.4310, -0.3480, 0.0702,  -0.1558};

    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
    std::vector<float> bias_data{
        0.0560,  0.0310, -0.1669, -0.0781, 0.1793, -0.1758, 0.3173,  -0.1650, -0.3732, 0.2946,
        -0.0912, 0.3118, 0.1391,  0.2755,  0.2695, -0.1059, -0.2357, 0.3629,  -0.2534, -0.0494,
        0.0556,  0.0881, -0.2592, -0.2213, 0.2310, -0.4044, 0.1801,  0.1438,  0.3108,  -0.3607};

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    std::vector<float> input{-0.8432,
                             -0.9887,
                             1.3041,
                             -2.6430,
                             -0.3306,
                             -0.8504,
                             -0.3933,
                             0.5151,
                             -0.2951,
                             0.0093,
                             -1.1948,
                             -0.1239,
                             0.0373,
                             1.3211,
                             0.7854,
                             -0.4838,
                             -1.0536,
                             -0.2529};

    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    std::vector<float> ih_data{
        -0.0468, 0.5691, -0.0882, 0.8340, 0.1483, -0.3902, -0.5348, 0.4178, 1.0175, 0.9212};
    float clip = 0.0f;

    // no activation function specified, so default is used.
    {
        migraphx::program p;
        auto* mm       = p.get_main_module();
        auto seq       = mm->add_literal(migraphx::literal{in_shape, input});
        auto w         = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r         = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias      = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und       = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih        = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func", {}},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), concat_hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.3969709,
                                        0.43360898,
                                        0.35775262,
                                        0.23280787,
                                        -0.52179873,
                                        -0.21944991,
                                        0.4535257,
                                        -0.13735442,
                                        0.51757574,
                                        0.50380427};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // 1 activation function (sigmoid) specified
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(
                      std::vector<migraphx::operation>{migraphx::make_op("sigmoid")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{0.26905832, 0.5669211,  0.20464146, 0.67195725, 0.24752215,
                                        0.11411376, 0.12353572, 0.4245067,  0.73908687, 0.8644615,
                                        0.34754312, 0.61424744, 0.36769435, 0.6499579,  0.3168031,
                                        0.3296533,  0.3055136,  0.42514813, 0.6851256,  0.7967266,
                                        0.35652235, 0.6033026,  0.52634895, 0.5815402,  0.3001663,
                                        0.39814138, 0.4354002,  0.4310627,  0.6708563,  0.7509278};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // 1 activation function (tanh) specified
    {
        migraphx::program p;
        auto* mm       = p.get_main_module();
        auto seq       = mm->add_literal(migraphx::literal{in_shape, input});
        auto w         = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r         = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias      = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und       = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih        = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), concat_hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.49333298,
                                        -0.06104589,
                                        0.5629142,
                                        -0.97955984,
                                        -0.9314696,
                                        -0.03033514,
                                        0.5280315,
                                        -0.27354342,
                                        0.65615714,
                                        0.53612584};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // seq length of 1
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        seq_len  = 1;
        migraphx::shape in_shape_one{migraphx::shape::float_type,
                                     {seq_len, batch_size, input_size}};
        std::vector<float> input_one{-0.8432, -0.9887, 1.3041, -2.6430, -0.3306, -0.8504};
        auto seq  = mm->add_literal(migraphx::literal{in_shape_one, input_one});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);

        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.27298412,
                                        0.42363745,
                                        -0.09368783,
                                        0.4823072,
                                        -0.02183238,
                                        -0.6873896,
                                        0.16144305,
                                        0.31932795,
                                        0.6104771,
                                        0.79759157};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(gru_reverse)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, input_size}};
    std::vector<float> w_data{
        0.3485,  -0.0378, -0.1782, 0.1416,  -0.3096, -0.2212, -0.3883, 0.1983,  -0.2418,
        0.1480,  -0.3255, 0.1359,  -0.3551, -0.3605, -0.3482, -0.1424, -0.0495, -0.1640,
        -0.1979, -0.2577, -0.4097, -0.1211, -0.0412, 0.1801,  0.1721,  -0.4327, -0.0498,
        0.2628,  -0.1573, -0.1577, 0.2759,  -0.2023, -0.1185, -0.2136, 0.1294,  -0.2331,
        0.0701,  0.4316,  0.0480,  0.0247,  -0.0166, -0.2729, 0.1712,  -0.3984, -0.3905};

    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, hidden_size}};
    std::vector<float> r_data{
        0.2848,  -0.2851, -0.3466, -0.1718, -0.1492, -0.0082, 0.2452,  -0.0401, 0.3399,  0.2529,
        -0.0953, -0.0903, -0.1518, -0.1373, 0.3848,  -0.0130, -0.4339, 0.0406,  -0.1926, -0.1131,
        0.4285,  -0.0013, 0.2243,  0.2752,  0.1776,  -0.1720, 0.0822,  -0.0295, 0.1062,  -0.2721,
        -0.2736, -0.1826, 0.3541,  -0.4259, 0.2188,  0.0706,  0.3650,  0.3947,  0.2522,  0.2179,
        -0.0744, 0.2122,  -0.4346, 0.2760,  0.4076,  0.1183,  -0.1500, -0.1704, 0.3090,  -0.0706,
        -0.2442, 0.3021,  0.1680,  0.0783,  -0.3754, -0.3469, -0.2972, -0.0170, 0.4143,  0.3801,
        0.3852,  -0.1170, -0.2937, 0.2979,  -0.1357, 0.4257,  0.3884,  -0.2916, 0.1071,  0.0934,
        0.3645,  -0.4310, -0.3480, 0.0702,  -0.1558};

    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
    std::vector<float> bias_data{
        0.0560,  0.0310, -0.1669, -0.0781, 0.1793, -0.1758, 0.3173,  -0.1650, -0.3732, 0.2946,
        -0.0912, 0.3118, 0.1391,  0.2755,  0.2695, -0.1059, -0.2357, 0.3629,  -0.2534, -0.0494,
        0.0556,  0.0881, -0.2592, -0.2213, 0.2310, -0.4044, 0.1801,  0.1438,  0.3108,  -0.3607};

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    std::vector<float> input{-0.8432,
                             -0.9887,
                             1.3041,
                             -2.6430,
                             -0.3306,
                             -0.8504,
                             -0.3933,
                             0.5151,
                             -0.2951,
                             0.0093,
                             -1.1948,
                             -0.1239,
                             0.0373,
                             1.3211,
                             0.7854,
                             -0.4838,
                             -1.0536,
                             -0.2529};

    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    std::vector<float> ih_data{
        -0.0468, 0.5691, -0.0882, 0.8340, 0.1483, -0.3902, -0.5348, 0.4178, 1.0175, 0.9212};
    float clip = 0.0f;

    // concatenation of hidden states and last hs output for outputs
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto hs   = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                   {"actv_func",
                    migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                        migraphx::make_op("tanh")})},
                   {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                   {"clip", clip},
                   {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        mm->add_return({lho, hs});
        p.compile(migraphx::make_target("ref"));
        auto outputs = p.eval({});

        const auto& res_hs  = outputs.back();
        const auto& res_lho = outputs.front();
        std::vector<float> hs_data;
        std::vector<float> lho_data;
        res_hs.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });
        res_lho.visit([&](auto output) { lho_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.263403, 0.317655,  -0.00634162, 0.200443, -0.349125,
                                        -0.600874, 0.542386,  -0.0856531,  0.55703,  0.54711,
                                        -0.276245, 0.521348,  0.302874,    0.394353, -0.334369,
                                        -0.187861, 0.213553,  -0.0708377,  0.545435, 0.654301,
                                        -0.329512, 0.476095,  0.284044,    0.392077, -0.369226,
                                        -0.3275,   -0.027301, 0.143774,    0.655686, 0.782831};
        std::vector<float> lho_data_gold{-0.263403,
                                         0.317655,
                                         -0.00634162,
                                         0.200443,
                                         -0.349125,
                                         -0.600874,
                                         0.542386,
                                         -0.0856531,
                                         0.55703,
                                         0.54711};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(lho_data, lho_data_gold));
    }

    // variable input sequence length
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data{1, 2};
        auto sql = mm->add_literal(seq_len_s, len_data);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        mm->add_return({lho, hs});
        p.compile(migraphx::make_target("ref"));
        auto outputs = p.eval({});

        const auto& res_hs  = outputs.back();
        const auto& res_lho = outputs.front();
        std::vector<float> hs_data;
        std::vector<float> lho_data;
        res_hs.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });
        res_lho.visit([&](auto output) { lho_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            -0.272984,  0.423637,  -0.0936878, 0.482307, -0.0218324, -0.630874, 0.401448, 0.0488417,
            0.558397,   0.664423,  0,          0,        0,          0,         0,        -0.238202,
            -0.0752721, 0.0919409, 0.669654,   0.782363, 0,          0,         0,        0,
            0,          0,         0,          0,        0,          0};
        std::vector<float> lho_data_gold{-0.272984,
                                         0.423637,
                                         -0.0936878,
                                         0.482307,
                                         -0.0218324,
                                         -0.630874,
                                         0.401448,
                                         0.0488417,
                                         0.558397,
                                         0.664423};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(lho_data, lho_data_gold));
    }

    // last output for output, linear_before_reset = 0
    {
        migraphx::program p;
        auto* mm       = p.get_main_module();
        auto seq       = mm->add_literal(migraphx::literal{in_shape, input});
        auto w         = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r         = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias      = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und       = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih        = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"linear_before_reset", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), concat_hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.388654,
                                        0.384975,
                                        0.0179455,
                                        0.350101,
                                        -0.456872,
                                        -0.690085,
                                        0.534512,
                                        -0.0558191,
                                        0.646604,
                                        0.463943};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // no activation function specified, so default is used.
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func", {}},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.263403, 0.317655,  -0.00634162, 0.200443, -0.349125,
                                        -0.600874, 0.542386,  -0.0856531,  0.55703,  0.54711,
                                        -0.276245, 0.521348,  0.302874,    0.394353, -0.334369,
                                        -0.187861, 0.213553,  -0.0708377,  0.545435, 0.654301,
                                        -0.329512, 0.476095,  0.284044,    0.392077, -0.369226,
                                        -0.3275,   -0.027301, 0.143774,    0.655686, 0.782831};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // seq length of 1
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        seq_len  = 1;
        migraphx::shape in_shape_one{migraphx::shape::float_type,
                                     {seq_len, batch_size, input_size}};
        std::vector<float> input_one{-0.8432, -0.9887, 1.3041, -2.6430, -0.3306, -0.8504};
        auto seq  = mm->add_literal(migraphx::literal{in_shape_one, input_one});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);

        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.272984,
                                        0.423637,
                                        -0.0936878,
                                        0.482307,
                                        -0.0218324,
                                        -0.68739,
                                        0.161443,
                                        0.319328,
                                        0.610477,
                                        0.797592};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(gru_reverse_layout)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, input_size}};
    std::vector<float> w_data{
        0.3485,  -0.0378, -0.1782, 0.1416,  -0.3096, -0.2212, -0.3883, 0.1983,  -0.2418,
        0.1480,  -0.3255, 0.1359,  -0.3551, -0.3605, -0.3482, -0.1424, -0.0495, -0.1640,
        -0.1979, -0.2577, -0.4097, -0.1211, -0.0412, 0.1801,  0.1721,  -0.4327, -0.0498,
        0.2628,  -0.1573, -0.1577, 0.2759,  -0.2023, -0.1185, -0.2136, 0.1294,  -0.2331,
        0.0701,  0.4316,  0.0480,  0.0247,  -0.0166, -0.2729, 0.1712,  -0.3984, -0.3905};

    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, hidden_size}};
    std::vector<float> r_data{
        0.2848,  -0.2851, -0.3466, -0.1718, -0.1492, -0.0082, 0.2452,  -0.0401, 0.3399,  0.2529,
        -0.0953, -0.0903, -0.1518, -0.1373, 0.3848,  -0.0130, -0.4339, 0.0406,  -0.1926, -0.1131,
        0.4285,  -0.0013, 0.2243,  0.2752,  0.1776,  -0.1720, 0.0822,  -0.0295, 0.1062,  -0.2721,
        -0.2736, -0.1826, 0.3541,  -0.4259, 0.2188,  0.0706,  0.3650,  0.3947,  0.2522,  0.2179,
        -0.0744, 0.2122,  -0.4346, 0.2760,  0.4076,  0.1183,  -0.1500, -0.1704, 0.3090,  -0.0706,
        -0.2442, 0.3021,  0.1680,  0.0783,  -0.3754, -0.3469, -0.2972, -0.0170, 0.4143,  0.3801,
        0.3852,  -0.1170, -0.2937, 0.2979,  -0.1357, 0.4257,  0.3884,  -0.2916, 0.1071,  0.0934,
        0.3645,  -0.4310, -0.3480, 0.0702,  -0.1558};

    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
    std::vector<float> bias_data{
        0.0560,  0.0310, -0.1669, -0.0781, 0.1793, -0.1758, 0.3173,  -0.1650, -0.3732, 0.2946,
        -0.0912, 0.3118, 0.1391,  0.2755,  0.2695, -0.1059, -0.2357, 0.3629,  -0.2534, -0.0494,
        0.0556,  0.0881, -0.2592, -0.2213, 0.2310, -0.4044, 0.1801,  0.1438,  0.3108,  -0.3607};

    migraphx::shape in_shape{migraphx::shape::float_type, {batch_size, seq_len, input_size}};
    std::vector<float> input{-0.8432,
                             -0.9887,
                             1.3041,
                             -0.3933,
                             0.5151,
                             -0.2951,
                             0.0373,
                             1.3211,
                             0.7854,
                             -2.6430,
                             -0.3306,
                             -0.8504,
                             0.0093,
                             -1.1948,
                             -0.1239,
                             -0.4838,
                             -1.0536,
                             -0.2529};

    migraphx::shape ih_shape{migraphx::shape::float_type, {batch_size, num_dirct, hidden_size}};
    std::vector<float> ih_data{
        -0.0468, 0.5691, -0.0882, 0.8340, 0.1483, -0.3902, -0.5348, 0.4178, 1.0175, 0.9212};
    float clip = 0.0f;

    // concatenation of hidden states and last hs output for outputs
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        hs  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}), hs);
        lho = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), lho);
        mm->add_return({lho, hs});
        p.compile(migraphx::make_target("ref"));
        auto outputs = p.eval({});

        const auto& res_hs  = outputs.back();
        const auto& res_lho = outputs.front();
        std::vector<float> hs_data;
        std::vector<float> lho_data;
        res_hs.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });
        res_lho.visit([&](auto output) { lho_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.263403, 0.317655,  -0.00634162, 0.200443, -0.349125,
                                        -0.276245, 0.521348,  0.302874,    0.394353, -0.334369,
                                        -0.329512, 0.476095,  0.284044,    0.392077, -0.369226,
                                        -0.600874, 0.542386,  -0.0856531,  0.55703,  0.54711,
                                        -0.187861, 0.213553,  -0.0708377,  0.545435, 0.654301,
                                        -0.3275,   -0.027301, 0.143774,    0.655686, 0.782831};
        std::vector<float> lho_data_gold{-0.263403,
                                         0.317655,
                                         -0.00634162,
                                         0.200443,
                                         -0.349125,
                                         -0.600874,
                                         0.542386,
                                         -0.0856531,
                                         0.55703,
                                         0.54711};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(lho_data, lho_data_gold));
    }

    // variable input sequence length
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data{1, 2};
        auto sql = mm->add_literal(seq_len_s, len_data);

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        hs  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}), hs);
        lho = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), lho);
        mm->add_return({lho, hs});
        p.compile(migraphx::make_target("ref"));
        auto outputs = p.eval({});

        const auto& res_hs  = outputs.back();
        const auto& res_lho = outputs.front();
        std::vector<float> hs_data;
        std::vector<float> lho_data;
        res_hs.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });
        res_lho.visit([&](auto output) { lho_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.272984, 0.423637,   -0.0936878, 0.482307, -0.0218324,
                                        0,         0,          0,          0,        0,
                                        0,         0,          0,          0,        0,
                                        -0.630874, 0.401448,   0.0488417,  0.558397, 0.664423,
                                        -0.238202, -0.0752721, 0.0919409,  0.669654, 0.782363,
                                        0,         0,          0,          0,        0};
        std::vector<float> lho_data_gold{-0.272984,
                                         0.423637,
                                         -0.0936878,
                                         0.482307,
                                         -0.0218324,
                                         -0.630874,
                                         0.401448,
                                         0.0488417,
                                         0.558397,
                                         0.664423};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(lho_data, lho_data_gold));
    }

    // last output for output, linear_before_reset = 0
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto concat_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"linear_before_reset", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), concat_hs);
        lho = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), lho);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.388654,
                                        0.384975,
                                        0.0179455,
                                        0.350101,
                                        -0.456872,
                                        -0.690085,
                                        0.534512,
                                        -0.0558191,
                                        0.646604,
                                        0.463943};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // seq length of 1
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        seq_len  = 1;
        migraphx::shape in_shape_one{migraphx::shape::float_type,
                                     {batch_size, seq_len, input_size}};
        std::vector<float> input_one{-0.8432, -0.9887, 1.3041, -2.6430, -0.3306, -0.8504};
        auto seq  = mm->add_literal(migraphx::literal{in_shape_one, input_one});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}), hs);

        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.272984,
                                        0.423637,
                                        -0.0936878,
                                        0.482307,
                                        -0.0218324,
                                        -0.68739,
                                        0.161443,
                                        0.319328,
                                        0.610477,
                                        0.797592};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(gru_bidirectional)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 2;
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, input_size}};
    std::vector<float> w_data{
        0.3809,  0.4283,  0.2294,  -0.1018, -0.1226, -0.0037, 0.2449,  -0.2712, -0.1418,
        0.1363,  -0.3453, -0.0693, -0.2281, 0.2699,  -0.2024, -0.3085, -0.3338, 0.4109,
        0.2605,  -0.1019, -0.2813, 0.3323,  -0.1590, 0.0788,  -0.3535, 0.0397,  0.2732,
        0.2906,  0.0519,  0.3617,  -0.2664, 0.1441,  0.0464,  -0.1057, 0.2204,  -0.3294,
        0.3670,  0.1411,  0.3852,  0.3572,  0.3918,  0.0483,  -0.3906, -0.2841, -0.2778,

        -0.4272, 0.2335,  -0.1811, -0.3885, -0.1279, 0.1000,  0.0206,  -0.3284, -0.0353,
        0.1197,  0.1190,  0.3862,  0.0965,  -0.0492, 0.2657,  -0.1430, 0.0597,  0.1408,
        -0.0315, 0.1248,  0.0751,  0.3838,  0.3020,  0.0515,  0.2375,  -0.4255, 0.1714,
        -0.0432, 0.3447,  -0.2441, -0.3989, -0.3428, -0.4204, -0.4080, -0.2683, -0.0996,
        -0.1685, -0.0532, -0.1258, 0.1663,  -0.3526, -0.3915, -0.1721, 0.1292,  -0.2279};

    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, hidden_size}};
    std::vector<float> r_data{
        -0.2683, 0.0699,  -0.4021, -0.1379, 0.0042,  -0.2447, 0.4006,  0.0270,  -0.0446, 0.1063,
        0.1381,  0.1310,  -0.3596, 0.3869,  0.3929,  0.2750,  0.0890,  0.3069,  -0.1691, -0.2194,
        -0.1066, 0.3187,  -0.4369, -0.0603, -0.0834, -0.1182, -0.2047, 0.3253,  -0.2931, 0.2082,
        0.0424,  0.1111,  -0.2773, -0.0279, -0.0869, 0.1413,  -0.4227, -0.3672, 0.4137,  0.0609,
        0.4223,  -0.4032, 0.2945,  0.3600,  0.3345,  -0.3880, -0.0192, -0.0090, -0.2648, 0.4339,
        -0.0155, 0.4437,  -0.1766, 0.1957,  0.2475,  0.3773,  -0.2710, 0.3289,  -0.2077, -0.2534,
        -0.0832, -0.1632, 0.0728,  0.2520,  0.4153,  0.1659,  -0.4342, 0.0541,  0.1812,  -0.2305,
        0.4440,  0.0946,  0.0410,  -0.4381, -0.3161, 0.3906,  -0.3958, -0.4238, 0.1975,  0.3440,
        0.1437,  -0.0568, 0.1492,  -0.4248, -0.3304, 0.2786,  -0.1328, -0.3740, -0.3566, 0.3074,
        0.0924,  0.2684,  -0.1527, 0.1826,  0.2424,  0.2002,  0.3479,  -0.1089, 0.3472,  -0.3677,
        -0.4231, -0.0798, -0.3709, 0.3924,  0.2774,  -0.3690, -0.0233, 0.2845,  0.1969,  0.1618,
        -0.3742, -0.3619, 0.2925,  -0.1838, -0.1495, -0.3747, 0.0341,  -0.4243, -0.0732, -0.3997,
        0.2139,  0.2425,  0.4171,  -0.3358, 0.3534,  0.0938,  -0.0582, -0.2681, -0.4293, 0.1027,
        0.4101,  0.2641,  -0.4110, -0.1681, 0.3582,  -0.2089, 0.0852,  0.0963,  0.3866,  0.1955,
        -0.2174, 0.1996,  -0.2252, 0.1748,  0.1833,  -0.3155, 0.2567,  -0.4387, 0.3402,  0.0599};

    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
    std::vector<float> bias_data{
        -0.1582, -0.0826, 0.4008,  0.0118,  0.2511,  0.1900,  -0.2838, 0.2549,  -0.2484, 0.2363,
        -0.4083, -0.0295, -0.1161, 0.1211,  0.2509,  -0.1414, -0.2628, -0.2992, 0.1517,  0.1817,
        -0.2783, 0.3183,  -0.1629, -0.3108, -0.3418, 0.0411,  0.2203,  0.2187,  -0.2990, -0.0416,
        0.0209,  -0.1024, 0.4443,  -0.4420, -0.0330, -0.3591, -0.2990, 0.2167,  0.1395,  0.2317,
        0.1318,  0.1909,  -0.3615, 0.1953,  -0.2582, -0.2217, 0.3723,  0.1458,  0.2630,  -0.0377,
        0.1754,  0.0800,  -0.3964, -0.3247, 0.4219,  -0.0900, 0.3553,  0.2614,  -0.1298, -0.1124};

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    std::vector<float> input{-0.8432,
                             -0.9887,
                             1.3041,
                             -2.6430,
                             -0.3306,
                             -0.8504,
                             -0.3933,
                             0.5151,
                             -0.2951,
                             0.0093,
                             -1.1948,
                             -0.1239,
                             0.0373,
                             1.3211,
                             0.7854,
                             -0.4838,
                             -1.0536,
                             -0.2529};

    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    std::vector<float> ih_data{-0.0468, 0.5691,  -0.0882, 0.8340,  0.1483, -0.3902, -0.5348,
                               0.4178,  1.0175,  0.9212,  -0.0468, 0.5691, -0.0882, 0.8340,
                               0.1483,  -0.3902, -0.5348, 0.4178,  1.0175, 0.9212};

    float clip = 0.0f;

    // concatenation of hidden states and last hs output for outputs
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto hs   = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                   {"actv_func",
                    migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                        migraphx::make_op("tanh"),
                                                                        migraphx::make_op("sigmoid"),
                                                                        migraphx::make_op("tanh")})},
                   {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                   {"clip", clip},
                   {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        mm->add_return({hs, lho});
        p.compile(migraphx::make_target("ref"));
        auto outputs   = p.eval({});
        const auto& hs_concat = outputs.front();
        const auto& res_lho   = outputs.back();
        std::vector<float> hs_data;
        std::vector<float> lho_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });
        res_lho.visit([&](auto output) { lho_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.0352243, 0.0146756,  0.00570925, 0.152446,  0.208683,   0.214342,   -0.0454273,
            -0.135177, -0.0800739, 0.903659,   0.0248217, 0.435231,   -0.144448,  0.101531,
            -0.111305, 0.381317,   0.468983,   0.230557,  0.348021,   0.180229,   -0.0930435,
            0.174108,  -0.063834,  0.0909285,  0.22759,   -0.221983,  -0.139656,  -0.0938906,
            -0.247681, 0.69647,    -0.159396,  0.299061,  -0.116652,  0.238649,   0.109945,
            0.192866,  0.307073,   0.191113,   0.658287,  -0.0340374, -0.0959787, 0.0794681,
            0.241526,  0.321104,   0.00693533, -0.311839, -0.12802,   -0.16643,   -0.393849,
            0.648851,  -0.395918,  0.231694,   -0.160503, 0.383289,   0.0879262,  -0.0254665,
            0.079043,  0.322652,   0.752701,   0.243775};
        std::vector<float> lho_data_gold{-0.0959787, 0.0794681, 0.241526,  0.321104,  0.00693533,
                                         -0.311839,  -0.12802,  -0.16643,  -0.393849, 0.648851,
                                         0.0248217,  0.435231,  -0.144448, 0.101531,  -0.111305,
                                         0.381317,   0.468983,  0.230557,  0.348021,  0.180229};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(lho_data, lho_data_gold));
    }

    // same input sequence length, but shorter than max squence length
    {
        migraphx::program p;
        auto* mm      = p.get_main_module();
        auto seq_orig = mm->add_literal(migraphx::literal{in_shape, input});
        auto w        = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r        = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias     = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto ih       = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        migraphx::shape pad_seq_s{migraphx::shape::float_type, {2, batch_size, input_size}};
        std::vector<float> pad_data(pad_seq_s.elements(), 0.0f);
        auto seq_p = mm->add_literal(migraphx::literal{pad_seq_s, pad_data});
        auto seq = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), seq_orig, seq_p);
        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data(batch_size, static_cast<int32_t>(seq_len));
        auto sql = mm->add_literal(seq_len_s, len_data);

        auto concat_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), concat_hs);
        mm->add_return({concat_hs, lho});
        p.compile(migraphx::make_target("ref"));
        auto outputs   = p.eval({});
        const auto& hs_concat = outputs.front();
        const auto& res_lho   = outputs.back();
        std::vector<float> hs_data;
        std::vector<float> lho_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });
        res_lho.visit([&](auto output) { lho_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.0352244, 0.0146756,  0.00570924, 0.152446,  0.208683,   0.214342,   -0.0454273,
            -0.135177, -0.0800739, 0.903659,   0.0248217, 0.435231,   -0.144448,  0.101531,
            -0.111305, 0.381317,   0.468983,   0.230557,  0.348021,   0.180229,   -0.0930435,
            0.174108,  -0.063834,  0.0909285,  0.22759,   -0.221983,  -0.139656,  -0.0938906,
            -0.247681, 0.69647,    -0.159396,  0.299061,  -0.116652,  0.238649,   0.109945,
            0.192866,  0.307073,   0.191113,   0.658287,  -0.0340374, -0.0959787, 0.0794681,
            0.241526,  0.321104,   0.00693531, -0.311839, -0.12802,   -0.16643,   -0.393849,
            0.648851,  -0.395918,  0.231694,   -0.160503, 0.383289,   0.0879262,  -0.0254665,
            0.079043,  0.322652,   0.752701,   0.243775,  0,          0,          0,
            0,         0,          0,          0,         0,          0,          0,
            0,         0,          0,          0,         0,          0,          0,
            0,         0,          0,          0,         0,          0,          0,
            0,         0,          0,          0,         0,          0,          0,
            0,         0,          0,          0,         0,          0,          0,
            0,         0};
        std::vector<float> lho_data_gold{-0.0959787, 0.0794681, 0.241526,  0.321104,  0.00693531,
                                         -0.311839,  -0.12802,  -0.16643,  -0.393849, 0.648851,
                                         0.0248217,  0.435231,  -0.144448, 0.101531,  -0.111305,
                                         0.381317,   0.468983,  0.230557,  0.348021,  0.180229};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(lho_data, lho_data_gold));
    }

    // variable input sequence lengths
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data{1, 2};
        auto sql = mm->add_literal(seq_len_s, len_data);

        auto concat_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), concat_hs);
        mm->add_return({concat_hs, lho});
        p.compile(migraphx::make_target("ref"));
        auto outputs   = p.eval({});
        const auto& hs_concat = outputs.front();
        const auto& res_lho   = outputs.back();
        std::vector<float> hs_data;
        std::vector<float> lho_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });
        res_lho.visit([&](auto output) { lho_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.0352244,  0.0146756,  0.00570924, 0.152446,   0.208683,  0.214342,  -0.0454273,
            -0.135177,  -0.0800739, 0.903659,   -0.0271321, 0.624762,  -0.117084, 0.509115,
            -0.0175078, 0.182457,   0.304506,   0.313825,   0.397697,  0.300873,  0,
            0,          0,          0,          0,          -0.221983, -0.139656, -0.0938906,
            -0.247681,  0.69647,    0,          0,          0,         0,         0,
            -0.059911,  0.0552807,  0.306764,   0.794409,   0.194492,  0,         0,
            0,          0,          0,          0,          0,         0,         0,
            0,          0,          0,          0,          0,         0,         0,
            0,          0,          0,          0};
        std::vector<float> lho_data_gold{0.0352244,  0.0146756, 0.00570924, 0.152446,  0.208683,
                                         -0.221983,  -0.139656, -0.0938906, -0.247681, 0.69647,
                                         -0.0271321, 0.624762,  -0.117084,  0.509115,  -0.0175078,
                                         0.182457,   0.304506,  0.313825,   0.397697,  0.300873};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(lho_data, lho_data_gold));
    }

    // last output for output, linear_before_reset = 0
    {
        migraphx::program p;
        auto* mm       = p.get_main_module();
        auto seq       = mm->add_literal(migraphx::literal{in_shape, input});
        auto w         = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r         = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias      = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und       = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih        = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"linear_before_reset", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), concat_hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            -0.09280921, 0.18506107, 0.32247013, 0.17034212, -0.00115255, -0.29865006, -0.04513004,
            -0.10688055, -0.4767866, 0.6317833,  0.00286336, 0.53692746,  -0.00617076, 0.04564289,
            -0.18030001, 0.39584228, 0.53879917, 0.384983,   0.2759448,   0.11611474};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(gru_bidirectional_layout)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 2;
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, input_size}};
    std::vector<float> w_data{
        0.3809,  0.4283,  0.2294,  -0.1018, -0.1226, -0.0037, 0.2449,  -0.2712, -0.1418,
        0.1363,  -0.3453, -0.0693, -0.2281, 0.2699,  -0.2024, -0.3085, -0.3338, 0.4109,
        0.2605,  -0.1019, -0.2813, 0.3323,  -0.1590, 0.0788,  -0.3535, 0.0397,  0.2732,
        0.2906,  0.0519,  0.3617,  -0.2664, 0.1441,  0.0464,  -0.1057, 0.2204,  -0.3294,
        0.3670,  0.1411,  0.3852,  0.3572,  0.3918,  0.0483,  -0.3906, -0.2841, -0.2778,

        -0.4272, 0.2335,  -0.1811, -0.3885, -0.1279, 0.1000,  0.0206,  -0.3284, -0.0353,
        0.1197,  0.1190,  0.3862,  0.0965,  -0.0492, 0.2657,  -0.1430, 0.0597,  0.1408,
        -0.0315, 0.1248,  0.0751,  0.3838,  0.3020,  0.0515,  0.2375,  -0.4255, 0.1714,
        -0.0432, 0.3447,  -0.2441, -0.3989, -0.3428, -0.4204, -0.4080, -0.2683, -0.0996,
        -0.1685, -0.0532, -0.1258, 0.1663,  -0.3526, -0.3915, -0.1721, 0.1292,  -0.2279};

    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, hidden_size}};
    std::vector<float> r_data{
        -0.2683, 0.0699,  -0.4021, -0.1379, 0.0042,  -0.2447, 0.4006,  0.0270,  -0.0446, 0.1063,
        0.1381,  0.1310,  -0.3596, 0.3869,  0.3929,  0.2750,  0.0890,  0.3069,  -0.1691, -0.2194,
        -0.1066, 0.3187,  -0.4369, -0.0603, -0.0834, -0.1182, -0.2047, 0.3253,  -0.2931, 0.2082,
        0.0424,  0.1111,  -0.2773, -0.0279, -0.0869, 0.1413,  -0.4227, -0.3672, 0.4137,  0.0609,
        0.4223,  -0.4032, 0.2945,  0.3600,  0.3345,  -0.3880, -0.0192, -0.0090, -0.2648, 0.4339,
        -0.0155, 0.4437,  -0.1766, 0.1957,  0.2475,  0.3773,  -0.2710, 0.3289,  -0.2077, -0.2534,
        -0.0832, -0.1632, 0.0728,  0.2520,  0.4153,  0.1659,  -0.4342, 0.0541,  0.1812,  -0.2305,
        0.4440,  0.0946,  0.0410,  -0.4381, -0.3161, 0.3906,  -0.3958, -0.4238, 0.1975,  0.3440,
        0.1437,  -0.0568, 0.1492,  -0.4248, -0.3304, 0.2786,  -0.1328, -0.3740, -0.3566, 0.3074,
        0.0924,  0.2684,  -0.1527, 0.1826,  0.2424,  0.2002,  0.3479,  -0.1089, 0.3472,  -0.3677,
        -0.4231, -0.0798, -0.3709, 0.3924,  0.2774,  -0.3690, -0.0233, 0.2845,  0.1969,  0.1618,
        -0.3742, -0.3619, 0.2925,  -0.1838, -0.1495, -0.3747, 0.0341,  -0.4243, -0.0732, -0.3997,
        0.2139,  0.2425,  0.4171,  -0.3358, 0.3534,  0.0938,  -0.0582, -0.2681, -0.4293, 0.1027,
        0.4101,  0.2641,  -0.4110, -0.1681, 0.3582,  -0.2089, 0.0852,  0.0963,  0.3866,  0.1955,
        -0.2174, 0.1996,  -0.2252, 0.1748,  0.1833,  -0.3155, 0.2567,  -0.4387, 0.3402,  0.0599};

    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
    std::vector<float> bias_data{
        -0.1582, -0.0826, 0.4008,  0.0118,  0.2511,  0.1900,  -0.2838, 0.2549,  -0.2484, 0.2363,
        -0.4083, -0.0295, -0.1161, 0.1211,  0.2509,  -0.1414, -0.2628, -0.2992, 0.1517,  0.1817,
        -0.2783, 0.3183,  -0.1629, -0.3108, -0.3418, 0.0411,  0.2203,  0.2187,  -0.2990, -0.0416,
        0.0209,  -0.1024, 0.4443,  -0.4420, -0.0330, -0.3591, -0.2990, 0.2167,  0.1395,  0.2317,
        0.1318,  0.1909,  -0.3615, 0.1953,  -0.2582, -0.2217, 0.3723,  0.1458,  0.2630,  -0.0377,
        0.1754,  0.0800,  -0.3964, -0.3247, 0.4219,  -0.0900, 0.3553,  0.2614,  -0.1298, -0.1124};

    migraphx::shape in_shape{migraphx::shape::float_type, {batch_size, seq_len, input_size}};
    std::vector<float> input{-0.8432,
                             -0.9887,
                             1.3041,
                             -0.3933,
                             0.5151,
                             -0.2951,
                             0.0373,
                             1.3211,
                             0.7854,
                             -2.6430,
                             -0.3306,
                             -0.8504,
                             0.0093,
                             -1.1948,
                             -0.1239,
                             -0.4838,
                             -1.0536,
                             -0.2529};

    migraphx::shape ih_shape{migraphx::shape::float_type, {batch_size, num_dirct, hidden_size}};
    std::vector<float> ih_data{-0.0468, 0.5691,  -0.0882, 0.8340,  0.1483,  -0.0468, 0.5691,
                               -0.0882, 0.8340,  0.1483,  -0.3902, -0.5348, 0.4178,  1.0175,
                               0.9212,  -0.3902, -0.5348, 0.4178,  1.0175,  0.9212};

    float clip = 0.0f;

    // concatenation of hidden states and last hs output for outputs
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        hs  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}), hs);
        lho = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), lho);
        mm->add_return({hs, lho});
        p.compile(migraphx::make_target("ref"));
        auto outputs   = p.eval({});
        const auto& hs_concat = outputs.front();
        const auto& res_lho   = outputs.back();
        std::vector<float> hs_data;
        std::vector<float> lho_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });
        res_lho.visit([&](auto output) { lho_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.0352243,  0.0146756, 0.00570925, 0.152446,   0.208683,  0.0248217,  0.435231,
            -0.144448,  0.101531,  -0.111305,  -0.0930435, 0.174108,  -0.063834,  0.0909285,
            0.22759,    -0.159396, 0.299061,   -0.116652,  0.238649,  0.109945,   -0.0959787,
            0.0794681,  0.241526,  0.321104,   0.00693533, -0.395918, 0.231694,   -0.160503,
            0.383289,   0.0879262, 0.214342,   -0.0454273, -0.135177, -0.0800739, 0.903659,
            0.381317,   0.468983,  0.230557,   0.348021,   0.180229,  -0.221983,  -0.139656,
            -0.0938906, -0.247681, 0.69647,    0.192866,   0.307073,  0.191113,   0.658287,
            -0.0340374, -0.311839, -0.12802,   -0.16643,   -0.393849, 0.648851,   -0.0254665,
            0.079043,   0.322652,  0.752701,   0.243775};

        std::vector<float> lho_data_gold{-0.0959787, 0.0794681, 0.241526,  0.321104,  0.00693533,
                                         0.0248217,  0.435231,  -0.144448, 0.101531,  -0.111305,
                                         -0.311839,  -0.12802,  -0.16643,  -0.393849, 0.648851,
                                         0.381317,   0.468983,  0.230557,  0.348021,  0.180229};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(lho_data, lho_data_gold));
    }

    // same input sequence length, but shorter than max squence length
    {
        migraphx::program p;
        auto* mm      = p.get_main_module();
        auto seq_orig = mm->add_literal(migraphx::literal{in_shape, input});
        auto w        = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r        = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias     = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto ih       = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        migraphx::shape pad_seq_s{migraphx::shape::float_type, {batch_size, 2, input_size}};
        std::vector<float> pad_data(pad_seq_s.elements(), 0.0f);
        auto seq_p = mm->add_literal(migraphx::literal{pad_seq_s, pad_data});
        auto seq = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), seq_orig, seq_p);
        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data(batch_size, static_cast<int32_t>(seq_len));
        auto sql = mm->add_literal(seq_len_s, len_data);

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto concat_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), concat_hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        concat_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                        concat_hs);
        lho = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), lho);
        mm->add_return({concat_hs, lho});
        p.compile(migraphx::make_target("ref"));
        auto outputs   = p.eval({});
        const auto& hs_concat = outputs.front();
        const auto& res_lho   = outputs.back();
        std::vector<float> hs_data;
        std::vector<float> lho_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });
        res_lho.visit([&](auto output) { lho_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.0352244, 0.0146756, 0.00570924, 0.152446,   0.208683,   0.0248217,  0.435231,
            -0.144448, 0.101531,  -0.111305,  -0.0930435, 0.174108,   -0.063834,  0.0909285,
            0.22759,   -0.159396, 0.299061,   -0.116652,  0.238649,   0.109945,   -0.0959787,
            0.0794681, 0.241526,  0.321104,   0.00693531, -0.395918,  0.231694,   -0.160503,
            0.383289,  0.0879262, 0,          0,          0,          0,          0,
            0,         0,         0,          0,          0,          0,          0,
            0,         0,         0,          0,          0,          0,          0,
            0,         0.214342,  -0.0454273, -0.135177,  -0.0800739, 0.903659,   0.381317,
            0.468983,  0.230557,  0.348021,   0.180229,   -0.221983,  -0.139656,  -0.0938906,
            -0.247681, 0.69647,   0.192866,   0.307073,   0.191113,   0.658287,   -0.0340374,
            -0.311839, -0.12802,  -0.16643,   -0.393849,  0.648851,   -0.0254665, 0.079043,
            0.322652,  0.752701,  0.243775,   0,          0,          0,          0,
            0,         0,         0,          0,          0,          0,          0,
            0,         0,         0,          0,          0,          0,          0,
            0,         0};

        std::vector<float> lho_data_gold{-0.0959787, 0.0794681, 0.241526,  0.321104,  0.00693531,
                                         0.0248217,  0.435231,  -0.144448, 0.101531,  -0.111305,
                                         -0.311839,  -0.12802,  -0.16643,  -0.393849, 0.648851,
                                         0.381317,   0.468983,  0.230557,  0.348021,  0.180229};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(lho_data, lho_data_gold));
    }

    // variable input sequence lengths
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data{1, 2};
        auto sql = mm->add_literal(seq_len_s, len_data);

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto concat_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), concat_hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        concat_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                        concat_hs);
        lho = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), lho);
        mm->add_return({concat_hs, lho});
        p.compile(migraphx::make_target("ref"));
        auto outputs   = p.eval({});
        const auto& hs_concat = outputs.front();
        const auto& res_lho   = outputs.back();
        std::vector<float> hs_data;
        std::vector<float> lho_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });
        res_lho.visit([&](auto output) { lho_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.0352244,  0.0146756, 0.00570924, 0.152446,   0.208683,  -0.0271321, 0.624762,
            -0.117084,  0.509115,  -0.0175078, 0,          0,         0,          0,
            0,          0,         0,          0,          0,         0,          0,
            0,          0,         0,          0,          0,         0,          0,
            0,          0,         0.214342,   -0.0454273, -0.135177, -0.0800739, 0.903659,
            0.182457,   0.304506,  0.313825,   0.397697,   0.300873,  -0.221983,  -0.139656,
            -0.0938906, -0.247681, 0.69647,    -0.059911,  0.0552807, 0.306764,   0.794409,
            0.194492,   0,         0,          0,          0,         0,          0,
            0,          0,         0,          0};
        std::vector<float> lho_data_gold{0.0352244,  0.0146756, 0.00570924, 0.152446,  0.208683,
                                         -0.0271321, 0.624762,  -0.117084,  0.509115,  -0.0175078,
                                         -0.221983,  -0.139656, -0.0938906, -0.247681, 0.69647,
                                         0.182457,   0.304506,  0.313825,   0.397697,  0.300873};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(lho_data, lho_data_gold));
    }

    // last output for output, linear_before_reset = 0
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto concat_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"linear_before_reset", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), concat_hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), lho);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            -0.09280921, 0.18506107, 0.32247013,  0.17034212,  -0.00115255, 0.00286336,  0.53692746,
            -0.00617076, 0.04564289, -0.18030001, -0.29865006, -0.04513004, -0.10688055, -0.4767866,
            0.6317833,   0.39584228, 0.53879917,  0.384983,    0.2759448,   0.11611474};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(gru_bidirectional_args)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 2;
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, input_size}};
    std::vector<float> w_data{
        0.3809,  0.4283,  0.2294,  -0.1018, -0.1226, -0.0037, 0.2449,  -0.2712, -0.1418,
        0.1363,  -0.3453, -0.0693, -0.2281, 0.2699,  -0.2024, -0.3085, -0.3338, 0.4109,
        0.2605,  -0.1019, -0.2813, 0.3323,  -0.1590, 0.0788,  -0.3535, 0.0397,  0.2732,
        0.2906,  0.0519,  0.3617,  -0.2664, 0.1441,  0.0464,  -0.1057, 0.2204,  -0.3294,
        0.3670,  0.1411,  0.3852,  0.3572,  0.3918,  0.0483,  -0.3906, -0.2841, -0.2778,

        -0.4272, 0.2335,  -0.1811, -0.3885, -0.1279, 0.1000,  0.0206,  -0.3284, -0.0353,
        0.1197,  0.1190,  0.3862,  0.0965,  -0.0492, 0.2657,  -0.1430, 0.0597,  0.1408,
        -0.0315, 0.1248,  0.0751,  0.3838,  0.3020,  0.0515,  0.2375,  -0.4255, 0.1714,
        -0.0432, 0.3447,  -0.2441, -0.3989, -0.3428, -0.4204, -0.4080, -0.2683, -0.0996,
        -0.1685, -0.0532, -0.1258, 0.1663,  -0.3526, -0.3915, -0.1721, 0.1292,  -0.2279};

    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, hidden_size}};
    std::vector<float> r_data{
        -0.2683, 0.0699,  -0.4021, -0.1379, 0.0042,  -0.2447, 0.4006,  0.0270,  -0.0446, 0.1063,
        0.1381,  0.1310,  -0.3596, 0.3869,  0.3929,  0.2750,  0.0890,  0.3069,  -0.1691, -0.2194,
        -0.1066, 0.3187,  -0.4369, -0.0603, -0.0834, -0.1182, -0.2047, 0.3253,  -0.2931, 0.2082,
        0.0424,  0.1111,  -0.2773, -0.0279, -0.0869, 0.1413,  -0.4227, -0.3672, 0.4137,  0.0609,
        0.4223,  -0.4032, 0.2945,  0.3600,  0.3345,  -0.3880, -0.0192, -0.0090, -0.2648, 0.4339,
        -0.0155, 0.4437,  -0.1766, 0.1957,  0.2475,  0.3773,  -0.2710, 0.3289,  -0.2077, -0.2534,
        -0.0832, -0.1632, 0.0728,  0.2520,  0.4153,  0.1659,  -0.4342, 0.0541,  0.1812,  -0.2305,
        0.4440,  0.0946,  0.0410,  -0.4381, -0.3161, 0.3906,  -0.3958, -0.4238, 0.1975,  0.3440,
        0.1437,  -0.0568, 0.1492,  -0.4248, -0.3304, 0.2786,  -0.1328, -0.3740, -0.3566, 0.3074,
        0.0924,  0.2684,  -0.1527, 0.1826,  0.2424,  0.2002,  0.3479,  -0.1089, 0.3472,  -0.3677,
        -0.4231, -0.0798, -0.3709, 0.3924,  0.2774,  -0.3690, -0.0233, 0.2845,  0.1969,  0.1618,
        -0.3742, -0.3619, 0.2925,  -0.1838, -0.1495, -0.3747, 0.0341,  -0.4243, -0.0732, -0.3997,
        0.2139,  0.2425,  0.4171,  -0.3358, 0.3534,  0.0938,  -0.0582, -0.2681, -0.4293, 0.1027,
        0.4101,  0.2641,  -0.4110, -0.1681, 0.3582,  -0.2089, 0.0852,  0.0963,  0.3866,  0.1955,
        -0.2174, 0.1996,  -0.2252, 0.1748,  0.1833,  -0.3155, 0.2567,  -0.4387, 0.3402,  0.0599};

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    std::vector<float> input{-0.8432,
                             -0.9887,
                             1.3041,
                             -2.6430,
                             -0.3306,
                             -0.8504,
                             -0.3933,
                             0.5151,
                             -0.2951,
                             0.0093,
                             -1.1948,
                             -0.1239,
                             0.0373,
                             1.3211,
                             0.7854,
                             -0.4838,
                             -1.0536,
                             -0.2529};

    float clip = 0.0f;

    // 3 args
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input});
        auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});
        mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"linear_before_reset", 0}}),
            seq,
            w,
            r);

        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.0863793,  -0.227845,  0.0283059, -0.258645, 0.14187,    0.43541,     0.190748,
            -0.530196,  -0.440444,  0.293767,  0.0402142, 0.0788687,  -0.013,      -0.233298,
            -0.0739615, 0.467104,   0.446285,  0.306097,  0.125636,   0.272524,    0.0949838,
            0.0522264,  -0.0872712, -0.084203, 0.140013,  0.12739,    -0.0111171,  -0.431119,
            -0.468382,  0.388067,   -0.109174, -0.119064, -0.0242958, -0.180555,   0.118983,
            0.341578,   0.275472,   0.0853083, 0.332205,  -0.0498387, 0.140338,    0.0319435,
            0.247019,   0.275848,   -0.158223, 0.0495464, -0.0681034, -0.418158,   -0.523234,
            0.469122,   -0.306578,  -0.221095, -0.106449, -0.248934,  -0.00682121, 0.288407,
            0.198708,   0.0695644,  0.211621,  0.00246037};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // 4 args (bias is used)
    {
        std::vector<float> bias_data{
            -0.1582, -0.0826, 0.4008,  0.0118,  0.2511,  0.1900,  -0.2838, 0.2549,  -0.2484,
            0.2363,  -0.4083, -0.0295, -0.1161, 0.1211,  0.2509,  -0.1414, -0.2628, -0.2992,
            0.1517,  0.1817,  -0.2783, 0.3183,  -0.1629, -0.3108, -0.3418, 0.0411,  0.2203,
            0.2187,  -0.2990, -0.0416, 0.0209,  -0.1024, 0.4443,  -0.4420, -0.0330, -0.3591,
            -0.2990, 0.2167,  0.1395,  0.2317,  0.1318,  0.1909,  -0.3615, 0.1953,  -0.2582,
            -0.2217, 0.3723,  0.1458,  0.2630,  -0.0377, 0.1754,  0.0800,  -0.3964, -0.3247,
            0.4219,  -0.0900, 0.3553,  0.2614,  -0.1298, -0.1124};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias);

        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            -0.156667, -0.248473,  0.0255282,  -0.24566,  0.211589,   0.192707,   0.253025,
            -0.515283, -0.414174,  0.227127,   0.124773,  0.284532,   -0.203929,  -0.120517,
            -0.2794,   0.547635,   0.518549,   0.0447674, 0.258461,   0.0502881,  -0.219516,
            0.0927382, -0.0760062, -0.0906231, 0.237615,  -0.215638,  0.0128074,  -0.425813,
            -0.433378, 0.375383,   -0.0381738, 0.117793,  -0.180851,  -0.0841245, -0.116649,
            0.419469,  0.393515,   -0.076395,  0.427436,  -0.264071,  -0.185829,  0.0483585,
            0.242955,  0.25233,    0.0148512,  -0.304127, -0.0616653, -0.411568,  -0.491748,
            0.476508,  -0.313413,  -0.0361821, -0.173037, -0.235731,  -0.163113,  0.349008,
            0.248674,  -0.0295413, 0.291437,   -0.165005};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // 4 args (ih is used)
    {
        std::vector<float> ih_data{-0.0468, 0.5691,  -0.0882, 0.8340,  0.1483, -0.3902, -0.5348,
                                   0.4178,  1.0175,  0.9212,  -0.0468, 0.5691, -0.0882, 0.8340,
                                   0.1483,  -0.3902, -0.5348, 0.4178,  1.0175, 0.9212};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input});
        auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto ih  = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto und = mm->add_instruction(migraphx::make_op("undefined"));
        mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            und,
            und,
            ih);

        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.248571,   0.0982155,  0.00808877, 0.0986508,   0.0969705, 0.434692,  -0.141696,
            -0.164271,  -0.121157,  0.863222,   -0.0718357,  0.137711,  0.109221,  -0.00207995,
            0.0331223,  0.262705,   0.346587,   0.457158,    0.240744,  0.404261,  0.222779,
            0.179757,   -0.0845316, 0.0690347,  0.10204,     0.100155,  -0.190286, -0.122062,
            -0.274379,  0.547281,   -0.226753,  -0.0397069,  0.120404,  0.171299,  0.259989,
            0.0864604,  0.111322,   0.331784,   0.604653,    0.181017,  0.237426,  0.0911999,
            0.233106,   0.32996,    -0.17175,   0.0190231,   -0.154805, -0.205631, -0.405354,
            0.519054,   -0.380409,  -0.0350301, -0.00633752, 0.403791,  0.181883,  -0.0977917,
            -0.0339407, 0.413089,   0.721238,   0.431879};
        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(gru_bidirectional_actv_funcs)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 2;
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, input_size}};
    std::vector<float> w_data{
        0.3809,  0.4283,  0.2294,  -0.1018, -0.1226, -0.0037, 0.2449,  -0.2712, -0.1418,
        0.1363,  -0.3453, -0.0693, -0.2281, 0.2699,  -0.2024, -0.3085, -0.3338, 0.4109,
        0.2605,  -0.1019, -0.2813, 0.3323,  -0.1590, 0.0788,  -0.3535, 0.0397,  0.2732,
        0.2906,  0.0519,  0.3617,  -0.2664, 0.1441,  0.0464,  -0.1057, 0.2204,  -0.3294,
        0.3670,  0.1411,  0.3852,  0.3572,  0.3918,  0.0483,  -0.3906, -0.2841, -0.2778,

        -0.4272, 0.2335,  -0.1811, -0.3885, -0.1279, 0.1000,  0.0206,  -0.3284, -0.0353,
        0.1197,  0.1190,  0.3862,  0.0965,  -0.0492, 0.2657,  -0.1430, 0.0597,  0.1408,
        -0.0315, 0.1248,  0.0751,  0.3838,  0.3020,  0.0515,  0.2375,  -0.4255, 0.1714,
        -0.0432, 0.3447,  -0.2441, -0.3989, -0.3428, -0.4204, -0.4080, -0.2683, -0.0996,
        -0.1685, -0.0532, -0.1258, 0.1663,  -0.3526, -0.3915, -0.1721, 0.1292,  -0.2279};

    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, hidden_size}};
    std::vector<float> r_data{
        -0.2683, 0.0699,  -0.4021, -0.1379, 0.0042,  -0.2447, 0.4006,  0.0270,  -0.0446, 0.1063,
        0.1381,  0.1310,  -0.3596, 0.3869,  0.3929,  0.2750,  0.0890,  0.3069,  -0.1691, -0.2194,
        -0.1066, 0.3187,  -0.4369, -0.0603, -0.0834, -0.1182, -0.2047, 0.3253,  -0.2931, 0.2082,
        0.0424,  0.1111,  -0.2773, -0.0279, -0.0869, 0.1413,  -0.4227, -0.3672, 0.4137,  0.0609,
        0.4223,  -0.4032, 0.2945,  0.3600,  0.3345,  -0.3880, -0.0192, -0.0090, -0.2648, 0.4339,
        -0.0155, 0.4437,  -0.1766, 0.1957,  0.2475,  0.3773,  -0.2710, 0.3289,  -0.2077, -0.2534,
        -0.0832, -0.1632, 0.0728,  0.2520,  0.4153,  0.1659,  -0.4342, 0.0541,  0.1812,  -0.2305,
        0.4440,  0.0946,  0.0410,  -0.4381, -0.3161, 0.3906,  -0.3958, -0.4238, 0.1975,  0.3440,
        0.1437,  -0.0568, 0.1492,  -0.4248, -0.3304, 0.2786,  -0.1328, -0.3740, -0.3566, 0.3074,
        0.0924,  0.2684,  -0.1527, 0.1826,  0.2424,  0.2002,  0.3479,  -0.1089, 0.3472,  -0.3677,
        -0.4231, -0.0798, -0.3709, 0.3924,  0.2774,  -0.3690, -0.0233, 0.2845,  0.1969,  0.1618,
        -0.3742, -0.3619, 0.2925,  -0.1838, -0.1495, -0.3747, 0.0341,  -0.4243, -0.0732, -0.3997,
        0.2139,  0.2425,  0.4171,  -0.3358, 0.3534,  0.0938,  -0.0582, -0.2681, -0.4293, 0.1027,
        0.4101,  0.2641,  -0.4110, -0.1681, 0.3582,  -0.2089, 0.0852,  0.0963,  0.3866,  0.1955,
        -0.2174, 0.1996,  -0.2252, 0.1748,  0.1833,  -0.3155, 0.2567,  -0.4387, 0.3402,  0.0599};

    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
    std::vector<float> bias_data{
        -0.1582, -0.0826, 0.4008,  0.0118,  0.2511,  0.1900,  -0.2838, 0.2549,  -0.2484, 0.2363,
        -0.4083, -0.0295, -0.1161, 0.1211,  0.2509,  -0.1414, -0.2628, -0.2992, 0.1517,  0.1817,
        -0.2783, 0.3183,  -0.1629, -0.3108, -0.3418, 0.0411,  0.2203,  0.2187,  -0.2990, -0.0416,
        0.0209,  -0.1024, 0.4443,  -0.4420, -0.0330, -0.3591, -0.2990, 0.2167,  0.1395,  0.2317,
        0.1318,  0.1909,  -0.3615, 0.1953,  -0.2582, -0.2217, 0.3723,  0.1458,  0.2630,  -0.0377,
        0.1754,  0.0800,  -0.3964, -0.3247, 0.4219,  -0.0900, 0.3553,  0.2614,  -0.1298, -0.1124};

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    std::vector<float> input{-0.8432,
                             -0.9887,
                             1.3041,
                             -2.6430,
                             -0.3306,
                             -0.8504,
                             -0.3933,
                             0.5151,
                             -0.2951,
                             0.0093,
                             -1.1948,
                             -0.1239,
                             0.0373,
                             1.3211,
                             0.7854,
                             -0.4838,
                             -1.0536,
                             -0.2529};

    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    std::vector<float> ih_data{-0.0468, 0.5691,  -0.0882, 0.8340,  0.1483, -0.3902, -0.5348,
                               0.4178,  1.0175,  0.9212,  -0.0468, 0.5691, -0.0882, 0.8340,
                               0.1483,  -0.3902, -0.5348, 0.4178,  1.0175, 0.9212};

    float clip = 0.0f;

    // no activation function specified, so default is used.
    {
        migraphx::program p;
        auto* mm       = p.get_main_module();
        auto seq       = mm->add_literal(migraphx::literal{in_shape, input});
        auto w         = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r         = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias      = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und       = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih        = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func", {}},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), concat_hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.0959787, 0.0794681, 0.241526,  0.321104,  0.00693533,
                                        -0.311839,  -0.12802,  -0.16643,  -0.393849, 0.648851,
                                        0.0248217,  0.435231,  -0.144448, 0.101531,  -0.111305,
                                        0.381317,   0.468983,  0.230557,  0.348021,  0.180229};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // 1 activation function (sigmoid) specified
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(
                      std::vector<migraphx::operation>{migraphx::make_op("sigmoid")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"linear_before_reset", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.325495, 0.469214, 0.164517, 0.585327, 0.328398, 0.457928, 0.065011, 0.35986,
            0.545029, 0.859425, 0.427923, 0.667133, 0.41591,  0.540971, 0.365475, 0.482058,
            0.565495, 0.556993, 0.607649, 0.543627, 0.428915, 0.537405, 0.306046, 0.518399,
            0.403561, 0.410694, 0.301163, 0.407397, 0.471334, 0.726446, 0.309389, 0.612072,
            0.360619, 0.590861, 0.366545, 0.367001, 0.433829, 0.501275, 0.72481,  0.512745,
            0.463795, 0.539649, 0.487682, 0.554471, 0.395916, 0.430744, 0.415923, 0.424275,
            0.409655, 0.698256, 0.126883, 0.554374, 0.216137, 0.671491, 0.263833, 0.0678646,
            0.132732, 0.477083, 0.802206, 0.626802};
        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // 2 activation functions (tanh) specified
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.0919632, -0.398302,   -0.0267752, -0.326771,  0.401983,  0.949841,   0.557779,
            -0.745259, -1.52726,    0.946066,   0.330446,   0.301982,  -0.443763,  -0.0655817,
            -0.326473, 0.861394,    0.560799,   -0.101768,  0.145142,  0.128956,   -0.329758,
            0.458253,  -0.339208,   0.289109,   0.36728,    -1.09574,  -0.181394,  -0.575781,
            -0.823083, 0.804262,    -0.0965933, 0.20405,    -0.430215, 0.00884668, 0.0716857,
            0.844222,  0.516472,    -0.191571,  0.596968,   -0.545405, -0.336693,  -0.0280516,
            0.339058,  1.00367,     0.12655,    -0.0984504, -0.174945, -0.5365,    0.183188,
            0.66716,   -0.704461,   -0.393346,  -0.627123,  0.210395,  0.0563026,  0.31419,
            0.759629,  0.000258222, 0.350835,   -0.682684};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // 3 activation functions specified
    {
        migraphx::program p;
        auto* mm       = p.get_main_module();
        auto seq       = mm->add_literal(migraphx::literal{in_shape, input});
        auto w         = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r         = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias      = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und       = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih        = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), concat_hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{0.351019, 0.474363, 0.570719,  0.717703,   0.468843,
                                        1.15142,  0.457633, 0.300962,  0.361245,   0.666199,
                                        0.330446, 0.301982, -0.443763, -0.0655817, -0.326473,
                                        0.861394, 0.560799, -0.101768, 0.145142,   0.128956};
        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // 4 activation functions all specified
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            und,
            ih);

        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.0352243, 0.0146756,  0.00570925, 0.152446,  0.208683,   0.214342,   -0.0454273,
            -0.135177, -0.0800739, 0.903659,   0.0248217, 0.435231,   -0.144448,  0.101531,
            -0.111305, 0.381317,   0.468983,   0.230557,  0.348021,   0.180229,   -0.0930435,
            0.174108,  -0.063834,  0.0909285,  0.22759,   -0.221983,  -0.139656,  -0.0938906,
            -0.247681, 0.69647,    -0.159396,  0.299061,  -0.116652,  0.238649,   0.109945,
            0.192866,  0.307073,   0.191113,   0.658287,  -0.0340374, -0.0959787, 0.0794681,
            0.241526,  0.321104,   0.00693533, -0.311839, -0.12802,   -0.16643,   -0.393849,
            0.648851,  -0.395918,  0.231694,   -0.160503, 0.383289,   0.0879262,  -0.0254665,
            0.079043,  0.322652,   0.752701,   0.243775};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(gru_bidirectional_seq_1)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 1;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 2;
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, input_size}};
    std::vector<float> w_data{
        0.3809,  0.4283,  0.2294,  -0.1018, -0.1226, -0.0037, 0.2449,  -0.2712, -0.1418,
        0.1363,  -0.3453, -0.0693, -0.2281, 0.2699,  -0.2024, -0.3085, -0.3338, 0.4109,
        0.2605,  -0.1019, -0.2813, 0.3323,  -0.1590, 0.0788,  -0.3535, 0.0397,  0.2732,
        0.2906,  0.0519,  0.3617,  -0.2664, 0.1441,  0.0464,  -0.1057, 0.2204,  -0.3294,
        0.3670,  0.1411,  0.3852,  0.3572,  0.3918,  0.0483,  -0.3906, -0.2841, -0.2778,

        -0.4272, 0.2335,  -0.1811, -0.3885, -0.1279, 0.1000,  0.0206,  -0.3284, -0.0353,
        0.1197,  0.1190,  0.3862,  0.0965,  -0.0492, 0.2657,  -0.1430, 0.0597,  0.1408,
        -0.0315, 0.1248,  0.0751,  0.3838,  0.3020,  0.0515,  0.2375,  -0.4255, 0.1714,
        -0.0432, 0.3447,  -0.2441, -0.3989, -0.3428, -0.4204, -0.4080, -0.2683, -0.0996,
        -0.1685, -0.0532, -0.1258, 0.1663,  -0.3526, -0.3915, -0.1721, 0.1292,  -0.2279};

    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, hidden_size}};
    std::vector<float> r_data{
        -0.2683, 0.0699,  -0.4021, -0.1379, 0.0042,  -0.2447, 0.4006,  0.0270,  -0.0446, 0.1063,
        0.1381,  0.1310,  -0.3596, 0.3869,  0.3929,  0.2750,  0.0890,  0.3069,  -0.1691, -0.2194,
        -0.1066, 0.3187,  -0.4369, -0.0603, -0.0834, -0.1182, -0.2047, 0.3253,  -0.2931, 0.2082,
        0.0424,  0.1111,  -0.2773, -0.0279, -0.0869, 0.1413,  -0.4227, -0.3672, 0.4137,  0.0609,
        0.4223,  -0.4032, 0.2945,  0.3600,  0.3345,  -0.3880, -0.0192, -0.0090, -0.2648, 0.4339,
        -0.0155, 0.4437,  -0.1766, 0.1957,  0.2475,  0.3773,  -0.2710, 0.3289,  -0.2077, -0.2534,
        -0.0832, -0.1632, 0.0728,  0.2520,  0.4153,  0.1659,  -0.4342, 0.0541,  0.1812,  -0.2305,
        0.4440,  0.0946,  0.0410,  -0.4381, -0.3161, 0.3906,  -0.3958, -0.4238, 0.1975,  0.3440,
        0.1437,  -0.0568, 0.1492,  -0.4248, -0.3304, 0.2786,  -0.1328, -0.3740, -0.3566, 0.3074,
        0.0924,  0.2684,  -0.1527, 0.1826,  0.2424,  0.2002,  0.3479,  -0.1089, 0.3472,  -0.3677,
        -0.4231, -0.0798, -0.3709, 0.3924,  0.2774,  -0.3690, -0.0233, 0.2845,  0.1969,  0.1618,
        -0.3742, -0.3619, 0.2925,  -0.1838, -0.1495, -0.3747, 0.0341,  -0.4243, -0.0732, -0.3997,
        0.2139,  0.2425,  0.4171,  -0.3358, 0.3534,  0.0938,  -0.0582, -0.2681, -0.4293, 0.1027,
        0.4101,  0.2641,  -0.4110, -0.1681, 0.3582,  -0.2089, 0.0852,  0.0963,  0.3866,  0.1955,
        -0.2174, 0.1996,  -0.2252, 0.1748,  0.1833,  -0.3155, 0.2567,  -0.4387, 0.3402,  0.0599};

    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
    std::vector<float> bias_data{
        -0.1582, -0.0826, 0.4008,  0.0118,  0.2511,  0.1900,  -0.2838, 0.2549,  -0.2484, 0.2363,
        -0.4083, -0.0295, -0.1161, 0.1211,  0.2509,  -0.1414, -0.2628, -0.2992, 0.1517,  0.1817,
        -0.2783, 0.3183,  -0.1629, -0.3108, -0.3418, 0.0411,  0.2203,  0.2187,  -0.2990, -0.0416,
        0.0209,  -0.1024, 0.4443,  -0.4420, -0.0330, -0.3591, -0.2990, 0.2167,  0.1395,  0.2317,
        0.1318,  0.1909,  -0.3615, 0.1953,  -0.2582, -0.2217, 0.3723,  0.1458,  0.2630,  -0.0377,
        0.1754,  0.0800,  -0.3964, -0.3247, 0.4219,  -0.0900, 0.3553,  0.2614,  -0.1298, -0.1124};

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    std::vector<float> input{-0.8432,
                             -0.9887,
                             1.3041,
                             -2.6430,
                             -0.3306,
                             -0.8504,
                             -0.3933,
                             0.5151,
                             -0.2951,
                             0.0093,
                             -1.1948,
                             -0.1239,
                             0.0373,
                             1.3211,
                             0.7854,
                             -0.4838,
                             -1.0536,
                             -0.2529};

    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    std::vector<float> ih_data{-0.0468, 0.5691,  -0.0882, 0.8340,  0.1483, -0.3902, -0.5348,
                               0.4178,  1.0175,  0.9212,  -0.0468, 0.5691, -0.0882, 0.8340,
                               0.1483,  -0.3902, -0.5348, 0.4178,  1.0175, 0.9212};

    float clip = 0.0f;
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape in_shape_one{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    std::vector<float> input_one{-0.8432, -0.9887, 1.3041, -2.6430, -0.3306, -0.8504};
    auto seq  = mm->add_literal(migraphx::literal{in_shape_one, input_one});
    auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
    auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
    auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
    auto und  = mm->add_instruction(migraphx::make_op("undefined"));
    auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
    mm->add_instruction(
        migraphx::make_op(
            "gru",
            {{"hidden_size", hidden_size},
             {"actv_func",
              migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                  migraphx::make_op("tanh"),
                                                                  migraphx::make_op("sigmoid"),
                                                                  migraphx::make_op("tanh")})},
             {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
             {"clip", clip},
             {"linear_before_reset", 1}}),
        seq,
        w,
        r,
        bias,
        und,
        ih);

    p.compile(migraphx::make_target("ref"));
    auto hs_concat = p.eval({}).back();
    std::vector<float> hs_data;
    hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

    std::vector<float> hs_data_gold{0.0352243,  0.0146756,  0.00570925, 0.152446,   0.208683,
                                    0.214342,   -0.0454273, -0.135177,  -0.0800739, 0.903659,
                                    -0.0271321, 0.624762,   -0.117084,  0.509115,   -0.0175078,
                                    -0.144492,  -0.0115366, 0.409153,   0.487015,   0.550755};

    EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
}

TEST_CASE(gru_fp16)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, input_size}};
    std::vector<float> w_data{
        0.3485,  -0.0378, -0.1782, 0.1416,  -0.3096, -0.2212, -0.3883, 0.1983,  -0.2418,
        0.1480,  -0.3255, 0.1359,  -0.3551, -0.3605, -0.3482, -0.1424, -0.0495, -0.1640,
        -0.1979, -0.2577, -0.4097, -0.1211, -0.0412, 0.1801,  0.1721,  -0.4327, -0.0498,
        0.2628,  -0.1573, -0.1577, 0.2759,  -0.2023, -0.1185, -0.2136, 0.1294,  -0.2331,
        0.0701,  0.4316,  0.0480,  0.0247,  -0.0166, -0.2729, 0.1712,  -0.3984, -0.3905};

    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, hidden_size}};
    std::vector<float> r_data{
        0.2848,  -0.2851, -0.3466, -0.1718, -0.1492, -0.0082, 0.2452,  -0.0401, 0.3399,  0.2529,
        -0.0953, -0.0903, -0.1518, -0.1373, 0.3848,  -0.0130, -0.4339, 0.0406,  -0.1926, -0.1131,
        0.4285,  -0.0013, 0.2243,  0.2752,  0.1776,  -0.1720, 0.0822,  -0.0295, 0.1062,  -0.2721,
        -0.2736, -0.1826, 0.3541,  -0.4259, 0.2188,  0.0706,  0.3650,  0.3947,  0.2522,  0.2179,
        -0.0744, 0.2122,  -0.4346, 0.2760,  0.4076,  0.1183,  -0.1500, -0.1704, 0.3090,  -0.0706,
        -0.2442, 0.3021,  0.1680,  0.0783,  -0.3754, -0.3469, -0.2972, -0.0170, 0.4143,  0.3801,
        0.3852,  -0.1170, -0.2937, 0.2979,  -0.1357, 0.4257,  0.3884,  -0.2916, 0.1071,  0.0934,
        0.3645,  -0.4310, -0.3480, 0.0702,  -0.1558};

    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
    std::vector<float> bias_data{
        0.0560,  0.0310, -0.1669, -0.0781, 0.1793, -0.1758, 0.3173,  -0.1650, -0.3732, 0.2946,
        -0.0912, 0.3118, 0.1391,  0.2755,  0.2695, -0.1059, -0.2357, 0.3629,  -0.2534, -0.0494,
        0.0556,  0.0881, -0.2592, -0.2213, 0.2310, -0.4044, 0.1801,  0.1438,  0.3108,  -0.3607};

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    std::vector<float> input{-0.8432,
                             -0.9887,
                             1.3041,
                             -2.6430,
                             -0.3306,
                             -0.8504,
                             -0.3933,
                             0.5151,
                             -0.2951,
                             0.0093,
                             -1.1948,
                             -0.1239,
                             0.0373,
                             1.3211,
                             0.7854,
                             -0.4838,
                             -1.0536,
                             -0.2529};

    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    std::vector<float> ih_data{
        -0.0468, 0.5691, -0.0882, 0.8340, 0.1483, -0.3902, -0.5348, 0.4178, 1.0175, 0.9212};
    float clip = 0.0f;

    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto seq  = mm->add_literal(migraphx::literal{in_shape, input});
    auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
    auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
    auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
    auto und  = mm->add_instruction(migraphx::make_op("undefined"));
    auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});

    auto seq_half = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), seq);
    auto w_half = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), w);
    auto r_half = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), r);
    auto bias_half = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), bias);
    auto und_half = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), und);
    auto ih_half = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), ih);

    mm->add_instruction(
        migraphx::make_op(
            "gru",
            {{"hidden_size", hidden_size},
             {"actv_func",
              migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                  migraphx::make_op("tanh"),
                                                                  migraphx::make_op("sigmoid"),
                                                                  migraphx::make_op("tanh")})},
             {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
             {"clip", clip},
             {"linear_before_reset", 1}}),
        seq_half,
        w_half,
        r_half,
        bias_half,
        und_half,
        ih_half);

    p.compile(migraphx::make_target("ref"));
    auto hs_concat = p.eval({}).back();
    std::vector<float> hs_data;
    hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

    std::vector<float> hs_data_gold{-0.27298412, 0.42363745, -0.09368783, 0.4823072,  -0.02183238,
                                    -0.6873896,  0.16144305, 0.31932795,  0.6104771,  0.79759157,
                                    -0.31791314, 0.5249062,  0.08800987,  0.46404213, -0.11872687,
                                    -0.26210734, 0.34448293, -0.0176422,  0.48523626, 0.60002893,
                                    -0.3969709,  0.43360898, 0.35775262,  0.23280787, -0.52179873,
                                    -0.21944991, 0.4535257,  -0.13735442, 0.51757574, 0.50380427};

    EXPECT(migraphx::verify::verify_range_with_tolerance(
        hs_data, migraphx::verify::expected{hs_data_gold}, migraphx::verify::tolerance{0.005}));
}

TEST_CASE(lstm_forward)
{
    std::size_t batch_size  = 3;
    std::size_t seq_len     = 4;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    std::vector<float> w_data{
        0.1236,  -0.3942, 0.4149,  0.0795,  0.4934,  -0.2858, 0.2602,  -0.3098, 0.0567,  0.3344,
        0.3607,  -0.0551, 0.4952,  0.3799,  0.0630,  -0.3532, 0.0023,  -0.0592, 0.4267,  0.2382,
        -0.0784, -0.0032, -0.2476, -0.0206, -0.4963, 0.4837,  0.0827,  0.0123,  -0.1203, -0.0279,
        -0.0049, 0.4721,  -0.3564, -0.1286, 0.4090,  -0.0504, 0.0575,  -0.2138, 0.1071,  0.1976,
        -0.0758, 0.0139,  -0.0761, 0.3991,  -0.2965, -0.4845, -0.1496, 0.3285};

    std::vector<float> r_data{
        0.1237,  0.1229,  -0.0766, -0.1144, -0.1186, 0.2922,  0.2478,  0.3159,  -0.0522, 0.1685,
        -0.4621, 0.1728,  0.0670,  -0.2458, -0.3835, -0.4589, -0.3109, 0.4908,  -0.0133, -0.1858,
        -0.0590, -0.0347, -0.2353, -0.0671, -0.3812, -0.0004, -0.1432, 0.2406,  0.1033,  -0.0265,
        -0.3902, 0.0755,  0.3733,  0.4383,  -0.3140, 0.2537,  -0.1818, -0.4127, 0.3506,  0.2562,
        0.2926,  0.1620,  -0.4849, -0.4861, 0.4426,  0.2106,  -0.0005, 0.4418,  -0.2926, -0.3100,
        0.1500,  -0.0362, -0.3801, -0.0065, -0.0631, 0.1277,  0.2315,  0.4087,  -0.3963, -0.4161,
        -0.2169, -0.1344, 0.3468,  -0.2260};

    std::vector<float> bias_data{0.0088,  0.1183,  0.1642,  -0.2631, -0.1330, -0.4008, 0.3881,
                                 -0.4407, -0.2760, 0.1274,  -0.0083, -0.2885, 0.3949,  -0.0182,
                                 0.4445,  0.3477,  0.2266,  0.3423,  -0.0674, -0.4067, 0.0807,
                                 0.1109,  -0.2036, 0.1782,  -0.2467, -0.0730, -0.4216, 0.0316,
                                 -0.3025, 0.3637,  -0.3181, -0.4655};

    std::vector<float> input_data{
        -0.5516, 0.2391, -1.6951, -0.4313, -0.9730, -0.2005, 2.3930,  -0.5221, -0.1331,
        -0.0910, 1.2122, -0.1952, 0.4661,  0.6494,  2.1332,  -1.0972, 0.9816,  0.1122,
        0.3577,  1.3508, -0.5366, 1.7449,  0.5483,  -0.0701, -0.4100, -2.2344, 0.3685,
        0.4583,  2.3794, 1.0372,  -0.8887, 0.7892,  -0.4012, -0.2818, -2.3374, 1.5310};

    std::vector<float> ih_data{1.9104,
                               -1.9004,
                               0.3337,
                               0.5741,
                               0.5671,
                               0.0458,
                               0.4514,
                               -0.8968,
                               -0.9201,
                               0.1962,
                               0.5771,
                               -0.5332};

    std::vector<float> ic_data{0.9569,
                               -0.5981,
                               1.1312,
                               1.0945,
                               1.1055,
                               -0.1212,
                               -0.9097,
                               0.7831,
                               -1.6991,
                               -1.9498,
                               -1.2567,
                               -0.4114};

    std::vector<float> pph_data{1.84369764,
                                0.68413646,
                                -0.44892886,
                                -1.50904413,
                                0.3860796,
                                -0.52186625,
                                1.08474445,
                                -1.80867321,
                                1.32594529,
                                0.4336262,
                                -0.83699064,
                                0.49162736};

    float clip = 0.0f;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape ic_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size}};

    // forward, hidden state concatenation as output
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih,
            ic,
            und);
        p.compile(migraphx::make_target("ref"));

        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.0417273, -0.272355,  0.206765,   0.223879,    0.138193,   -0.0322939, -0.0891815,
            0.15773,   0.19139,    -0.127708,  -0.409371,   -0.136186,  0.0742487,  -0.0800085,
            0.259897,  0.0670196,  0.184266,   0.0610048,   -0.138041,  0.0963885,  0.0213755,
            -0.146027, -0.0324509, -0.0620429, -0.00532985, 0.0440265,  0.29654,    -0.0463156,
            0.0498799, 0.125772,   0.0533032,  -0.131413,   0.0988431,  -0.018085,  -0.159434,
            0.030266,  -0.0847427, 0.0874114,  0.304256,    -0.0585745, -0.0223018, 0.131113,
            0.135643,  -0.0566208, 0.142701,   0.0342236,   -0.198664,  0.0702607};
        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // forward, last_output as program output
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih,
            ic,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        p.compile(migraphx::make_target("ref"));

        auto last_hs = p.eval({}).back();
        std::vector<float> output_data;
        last_hs.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });

        std::vector<float> output_data_gold{-0.0847427,
                                            0.0874114,
                                            0.304256,
                                            -0.0585745,
                                            -0.0223018,
                                            0.131113,
                                            0.135643,
                                            -0.0566208,
                                            0.142701,
                                            0.0342236,
                                            -0.198664,
                                            0.0702607};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // forward, last_cell_output as program output
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih,
            ic,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_cell_output"), hs);
        p.compile(migraphx::make_target("ref"));

        auto last_hs = p.eval({}).back();
        std::vector<float> output_data;
        last_hs.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });

        std::vector<float> output_data_gold{-0.111454,
                                            0.247794,
                                            0.471087,
                                            -0.220574,
                                            -0.048196,
                                            0.263184,
                                            0.283258,
                                            -0.14882,
                                            0.605585,
                                            0.078598,
                                            -0.64457,
                                            0.119811};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }
}

TEST_CASE(lstm_forward_layout)
{
    std::size_t batch_size  = 3;
    std::size_t seq_len     = 4;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;

    std::vector<float> w_data{
        0.1236,  -0.3942, 0.4149,  0.0795,  0.4934,  -0.2858, 0.2602,  -0.3098, 0.0567,  0.3344,
        0.3607,  -0.0551, 0.4952,  0.3799,  0.0630,  -0.3532, 0.0023,  -0.0592, 0.4267,  0.2382,
        -0.0784, -0.0032, -0.2476, -0.0206, -0.4963, 0.4837,  0.0827,  0.0123,  -0.1203, -0.0279,
        -0.0049, 0.4721,  -0.3564, -0.1286, 0.4090,  -0.0504, 0.0575,  -0.2138, 0.1071,  0.1976,
        -0.0758, 0.0139,  -0.0761, 0.3991,  -0.2965, -0.4845, -0.1496, 0.3285};

    std::vector<float> r_data{
        0.1237,  0.1229,  -0.0766, -0.1144, -0.1186, 0.2922,  0.2478,  0.3159,  -0.0522, 0.1685,
        -0.4621, 0.1728,  0.0670,  -0.2458, -0.3835, -0.4589, -0.3109, 0.4908,  -0.0133, -0.1858,
        -0.0590, -0.0347, -0.2353, -0.0671, -0.3812, -0.0004, -0.1432, 0.2406,  0.1033,  -0.0265,
        -0.3902, 0.0755,  0.3733,  0.4383,  -0.3140, 0.2537,  -0.1818, -0.4127, 0.3506,  0.2562,
        0.2926,  0.1620,  -0.4849, -0.4861, 0.4426,  0.2106,  -0.0005, 0.4418,  -0.2926, -0.3100,
        0.1500,  -0.0362, -0.3801, -0.0065, -0.0631, 0.1277,  0.2315,  0.4087,  -0.3963, -0.4161,
        -0.2169, -0.1344, 0.3468,  -0.2260};

    std::vector<float> bias_data{0.0088,  0.1183,  0.1642,  -0.2631, -0.1330, -0.4008, 0.3881,
                                 -0.4407, -0.2760, 0.1274,  -0.0083, -0.2885, 0.3949,  -0.0182,
                                 0.4445,  0.3477,  0.2266,  0.3423,  -0.0674, -0.4067, 0.0807,
                                 0.1109,  -0.2036, 0.1782,  -0.2467, -0.0730, -0.4216, 0.0316,
                                 -0.3025, 0.3637,  -0.3181, -0.4655};

    std::vector<float> input_data{
        -0.5516, 0.2391, -1.6951, -0.0910, 1.2122,  -0.1952, 0.3577,  1.3508,  -0.5366,
        0.4583,  2.3794, 1.0372,  -0.4313, -0.9730, -0.2005, 0.4661,  0.6494,  2.1332,
        1.7449,  0.5483, -0.0701, -0.8887, 0.7892,  -0.4012, 2.3930,  -0.5221, -0.1331,
        -1.0972, 0.9816, 0.1122,  -0.4100, -2.2344, 0.3685,  -0.2818, -2.3374, 1.5310};

    std::vector<float> ih_data{1.9104,
                               -1.9004,
                               0.3337,
                               0.5741,
                               0.5671,
                               0.0458,
                               0.4514,
                               -0.8968,
                               -0.9201,
                               0.1962,
                               0.5771,
                               -0.5332};

    std::vector<float> ic_data{0.9569,
                               -0.5981,
                               1.1312,
                               1.0945,
                               1.1055,
                               -0.1212,
                               -0.9097,
                               0.7831,
                               -1.6991,
                               -1.9498,
                               -1.2567,
                               -0.4114};

    std::vector<float> pph_data{1.84369764,
                                0.68413646,
                                -0.44892886,
                                -1.50904413,
                                0.3860796,
                                -0.52186625,
                                1.08474445,
                                -1.80867321,
                                1.32594529,
                                0.4336262,
                                -0.83699064,
                                0.49162736};

    float clip = 0.0f;
    migraphx::shape in_shape{migraphx::shape::float_type, {batch_size, seq_len, input_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {batch_size, num_dirct, hidden_size}};
    migraphx::shape ic_shape{migraphx::shape::float_type, {batch_size, num_dirct, hidden_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size}};

    // forward, hidden state concatenation as output
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);
        ic  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ic);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih,
            ic,
            und);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}), hs);
        p.compile(migraphx::make_target("ref"));

        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.0417273,  -0.272355,   0.206765,  0.223879,   0.0742487,  -0.0800085, 0.259897,
            0.0670196,  -0.00532985, 0.0440265, 0.29654,    -0.0463156, -0.0847427, 0.0874114,
            0.304256,   -0.0585745,  0.138193,  -0.0322939, -0.0891815, 0.15773,    0.184266,
            0.0610048,  -0.138041,   0.0963885, 0.0498799,  0.125772,   0.0533032,  -0.131413,
            -0.0223018, 0.131113,    0.135643,  -0.056620,  0.19139,    -0.127708,  -0.409371,
            -0.136186,  0.0213755,   -0.146027, -0.0324509, -0.0620429, 0.0988431,  -0.018085,
            -0.159434,  0.030266,    0.142701,  0.0342236,  -0.198664,  0.0702607};
        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // forward, last_output as program output
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);
        ic  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ic);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih,
            ic,
            und);
        auto last_output = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_output);
        p.compile(migraphx::make_target("ref"));

        auto last_hs = p.eval({}).back();
        std::vector<float> output_data;
        last_hs.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });

        std::vector<float> output_data_gold{-0.0847427,
                                            0.0874114,
                                            0.304256,
                                            -0.0585745,
                                            -0.0223018,
                                            0.131113,
                                            0.135643,
                                            -0.0566208,
                                            0.142701,
                                            0.0342236,
                                            -0.198664,
                                            0.0702607};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // forward, last_cell_output as program output
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);
        ic  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ic);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih,
            ic,
            und);
        auto cell_output = mm->add_instruction(migraphx::make_op("rnn_last_cell_output"), hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), cell_output);
        p.compile(migraphx::make_target("ref"));

        auto last_hs = p.eval({}).back();
        std::vector<float> output_data;
        last_hs.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });

        std::vector<float> output_data_gold{-0.111454,
                                            0.247794,
                                            0.471087,
                                            -0.220574,
                                            -0.048196,
                                            0.263184,
                                            0.283258,
                                            -0.14882,
                                            0.605585,
                                            0.078598,
                                            -0.64457,
                                            0.119811};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }
}

TEST_CASE(lstm_forward_more)
{
    std::size_t batch_size  = 3;
    std::size_t seq_len     = 4;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    std::vector<float> w_data{
        0.1236,  -0.3942, 0.4149,  0.0795,  0.4934,  -0.2858, 0.2602,  -0.3098, 0.0567,  0.3344,
        0.3607,  -0.0551, 0.4952,  0.3799,  0.0630,  -0.3532, 0.0023,  -0.0592, 0.4267,  0.2382,
        -0.0784, -0.0032, -0.2476, -0.0206, -0.4963, 0.4837,  0.0827,  0.0123,  -0.1203, -0.0279,
        -0.0049, 0.4721,  -0.3564, -0.1286, 0.4090,  -0.0504, 0.0575,  -0.2138, 0.1071,  0.1976,
        -0.0758, 0.0139,  -0.0761, 0.3991,  -0.2965, -0.4845, -0.1496, 0.3285};

    std::vector<float> r_data{
        0.1237,  0.1229,  -0.0766, -0.1144, -0.1186, 0.2922,  0.2478,  0.3159,  -0.0522, 0.1685,
        -0.4621, 0.1728,  0.0670,  -0.2458, -0.3835, -0.4589, -0.3109, 0.4908,  -0.0133, -0.1858,
        -0.0590, -0.0347, -0.2353, -0.0671, -0.3812, -0.0004, -0.1432, 0.2406,  0.1033,  -0.0265,
        -0.3902, 0.0755,  0.3733,  0.4383,  -0.3140, 0.2537,  -0.1818, -0.4127, 0.3506,  0.2562,
        0.2926,  0.1620,  -0.4849, -0.4861, 0.4426,  0.2106,  -0.0005, 0.4418,  -0.2926, -0.3100,
        0.1500,  -0.0362, -0.3801, -0.0065, -0.0631, 0.1277,  0.2315,  0.4087,  -0.3963, -0.4161,
        -0.2169, -0.1344, 0.3468,  -0.2260};

    std::vector<float> bias_data{0.0088,  0.1183,  0.1642,  -0.2631, -0.1330, -0.4008, 0.3881,
                                 -0.4407, -0.2760, 0.1274,  -0.0083, -0.2885, 0.3949,  -0.0182,
                                 0.4445,  0.3477,  0.2266,  0.3423,  -0.0674, -0.4067, 0.0807,
                                 0.1109,  -0.2036, 0.1782,  -0.2467, -0.0730, -0.4216, 0.0316,
                                 -0.3025, 0.3637,  -0.3181, -0.4655};

    std::vector<float> input_data{
        -0.5516, 0.2391, -1.6951, -0.4313, -0.9730, -0.2005, 2.3930,  -0.5221, -0.1331,
        -0.0910, 1.2122, -0.1952, 0.4661,  0.6494,  2.1332,  -1.0972, 0.9816,  0.1122,
        0.3577,  1.3508, -0.5366, 1.7449,  0.5483,  -0.0701, -0.4100, -2.2344, 0.3685,
        0.4583,  2.3794, 1.0372,  -0.8887, 0.7892,  -0.4012, -0.2818, -2.3374, 1.5310};

    std::vector<float> ih_data{1.9104,
                               -1.9004,
                               0.3337,
                               0.5741,
                               0.5671,
                               0.0458,
                               0.4514,
                               -0.8968,
                               -0.9201,
                               0.1962,
                               0.5771,
                               -0.5332};

    std::vector<float> ic_data{0.9569,
                               -0.5981,
                               1.1312,
                               1.0945,
                               1.1055,
                               -0.1212,
                               -0.9097,
                               0.7831,
                               -1.6991,
                               -1.9498,
                               -1.2567,
                               -0.4114};

    std::vector<float> pph_data{1.84369764,
                                0.68413646,
                                -0.44892886,
                                -1.50904413,
                                0.3860796,
                                -0.52186625,
                                1.08474445,
                                -1.80867321,
                                1.32594529,
                                0.4336262,
                                -0.83699064,
                                0.49162736};

    float clip = 0.0f;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape ic_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size}};

    // forward, 3 args
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});
        mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r);
        p.compile(migraphx::make_target("ref"));

        auto last_hs = p.eval({}).back();
        std::vector<float> output_data;
        last_hs.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });

        std::vector<float> output_data_gold{
            -0.0327039, -0.0543852, 0.114378,   -0.0768855, 0.0319021,  -0.00298698, -0.0623361,
            0.0598866,  0.101585,   0.0687269,  -0.161725,  -0.25617,   -0.0786602,  -0.0613048,
            0.179592,   -0.071286,  0.074206,   0.0124086,  -0.139544,  0.108016,    -0.00973633,
            -0.0552699, 0.0252681,  -0.0562072, -0.102509,  -0.0372696, 0.252296,    -0.144544,
            0.00496085, 0.0662588,  -0.048577,  -0.187329,  0.0855831,  -0.0171894,  -0.140202,
            0.0828391,  -0.165194,  -0.0372928, 0.273786,   -0.100877,  -0.0458544,  -0.0401315,
            0.0737483,  -0.064505,  0.136898,   0.00160891, -0.184812,  0.147774};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // forward, 8 args
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto pph  = mm->add_literal(migraphx::literal{pph_shape, pph_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih,
            ic,
            pph);
        p.compile(migraphx::make_target("ref"));

        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.079753,   -0.289854,  0.160043,   0.115056,  0.294074,   -0.0319677, -0.0955337,
            0.104168,   0.022618,   -0.121195,  -0.4065,   -0.252054,  0.186991,   -0.0624168,
            0.205513,   0.0836373,  0.421857,   0.0459771, -0.144955,  0.0720673,  -0.0300906,
            -0.0890598, -0.135266,  -0.0413375, 0.0459032, 0.0414126,  0.272303,   0.0393149,
            0.218258,   0.0944405,  0.0431211,  -0.132394, 0.103489,   0.0142918,  -0.123408,
            0.0401075,  -0.058052,  0.0795391,  0.266617,  -0.0128746, 0.0309878,  0.0971544,
            0.149294,   -0.0492549, 0.187761,   0.0501726, -0.121584,  0.0606723};
        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // forward, last_output as program output, sequence length shorter
    // than max_seq_len
    {
        migraphx::program p;
        auto* mm      = p.get_main_module();
        auto seq_orig = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w        = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r        = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias     = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto ih       = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic       = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        migraphx::shape pad_seq_s{migraphx::shape::float_type, {2, batch_size, input_size}};
        std::vector<float> pad_data(pad_seq_s.elements(), 0.0f);
        auto seq_p = mm->add_literal(migraphx::literal{pad_seq_s, pad_data});
        auto seq = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), seq_orig, seq_p);
        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data(batch_size, static_cast<int32_t>(seq_len));
        auto sql = mm->add_literal(seq_len_s, len_data);

        auto und = mm->add_instruction(migraphx::make_op("undefined"));

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih,
            ic,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        p.compile(migraphx::make_target("ref"));

        auto last_hs = p.eval({}).back();
        std::vector<float> output_data;
        last_hs.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });

        std::vector<float> output_data_gold{-0.0847427,
                                            0.0874114,
                                            0.304256,
                                            -0.0585745,
                                            -0.0223018,
                                            0.131113,
                                            0.135643,
                                            -0.0566208,
                                            0.142701,
                                            0.0342236,
                                            -0.198664,
                                            0.0702607};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // seq_len = 1
    {
        seq_len = 1;
        migraphx::shape in_shape1{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        std::vector<float> input_data1{
            -0.5516, 0.2391, -1.6951, -0.4313, -0.9730, -0.2005, 2.3930, -0.5221, -0.1331};
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape1, input_data1});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto pph  = mm->add_literal(migraphx::literal{pph_shape, pph_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih,
            ic,
            pph);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        p.compile(migraphx::make_target("ref"));

        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{0.079753,
                                        -0.289854,
                                        0.160043,
                                        0.115056,
                                        0.294074,
                                        -0.0319677,
                                        -0.0955337,
                                        0.104168,
                                        0.022618,
                                        -0.121195,
                                        -0.4065,
                                        -0.252054};
        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(lstm_forward_more_layout)
{
    std::size_t batch_size  = 3;
    std::size_t seq_len     = 4;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    std::vector<float> w_data{
        0.1236,  -0.3942, 0.4149,  0.0795,  0.4934,  -0.2858, 0.2602,  -0.3098, 0.0567,  0.3344,
        0.3607,  -0.0551, 0.4952,  0.3799,  0.0630,  -0.3532, 0.0023,  -0.0592, 0.4267,  0.2382,
        -0.0784, -0.0032, -0.2476, -0.0206, -0.4963, 0.4837,  0.0827,  0.0123,  -0.1203, -0.0279,
        -0.0049, 0.4721,  -0.3564, -0.1286, 0.4090,  -0.0504, 0.0575,  -0.2138, 0.1071,  0.1976,
        -0.0758, 0.0139,  -0.0761, 0.3991,  -0.2965, -0.4845, -0.1496, 0.3285};

    std::vector<float> r_data{
        0.1237,  0.1229,  -0.0766, -0.1144, -0.1186, 0.2922,  0.2478,  0.3159,  -0.0522, 0.1685,
        -0.4621, 0.1728,  0.0670,  -0.2458, -0.3835, -0.4589, -0.3109, 0.4908,  -0.0133, -0.1858,
        -0.0590, -0.0347, -0.2353, -0.0671, -0.3812, -0.0004, -0.1432, 0.2406,  0.1033,  -0.0265,
        -0.3902, 0.0755,  0.3733,  0.4383,  -0.3140, 0.2537,  -0.1818, -0.4127, 0.3506,  0.2562,
        0.2926,  0.1620,  -0.4849, -0.4861, 0.4426,  0.2106,  -0.0005, 0.4418,  -0.2926, -0.3100,
        0.1500,  -0.0362, -0.3801, -0.0065, -0.0631, 0.1277,  0.2315,  0.4087,  -0.3963, -0.4161,
        -0.2169, -0.1344, 0.3468,  -0.2260};

    std::vector<float> bias_data{0.0088,  0.1183,  0.1642,  -0.2631, -0.1330, -0.4008, 0.3881,
                                 -0.4407, -0.2760, 0.1274,  -0.0083, -0.2885, 0.3949,  -0.0182,
                                 0.4445,  0.3477,  0.2266,  0.3423,  -0.0674, -0.4067, 0.0807,
                                 0.1109,  -0.2036, 0.1782,  -0.2467, -0.0730, -0.4216, 0.0316,
                                 -0.3025, 0.3637,  -0.3181, -0.4655};

    std::vector<float> input_data{
        -0.5516, 0.2391, -1.6951, -0.0910, 1.2122,  -0.1952, 0.3577,  1.3508,  -0.5366,
        0.4583,  2.3794, 1.0372,  -0.4313, -0.9730, -0.2005, 0.4661,  0.6494,  2.1332,
        1.7449,  0.5483, -0.0701, -0.8887, 0.7892,  -0.4012, 2.3930,  -0.5221, -0.1331,
        -1.0972, 0.9816, 0.1122,  -0.4100, -2.2344, 0.3685,  -0.2818, -2.3374, 1.5310};

    std::vector<float> ih_data{1.9104,
                               -1.9004,
                               0.3337,
                               0.5741,
                               0.5671,
                               0.0458,
                               0.4514,
                               -0.8968,
                               -0.9201,
                               0.1962,
                               0.5771,
                               -0.5332};

    std::vector<float> ic_data{0.9569,
                               -0.5981,
                               1.1312,
                               1.0945,
                               1.1055,
                               -0.1212,
                               -0.9097,
                               0.7831,
                               -1.6991,
                               -1.9498,
                               -1.2567,
                               -0.4114};

    std::vector<float> pph_data{1.84369764,
                                0.68413646,
                                -0.44892886,
                                -1.50904413,
                                0.3860796,
                                -0.52186625,
                                1.08474445,
                                -1.80867321,
                                1.32594529,
                                0.4336262,
                                -0.83699064,
                                0.49162736};

    float clip = 0.0f;
    migraphx::shape in_shape{migraphx::shape::float_type, {batch_size, seq_len, input_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {batch_size, num_dirct, hidden_size}};
    migraphx::shape ic_shape{migraphx::shape::float_type, {batch_size, num_dirct, hidden_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size}};

    // forward, 3 args
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}), hs);
        p.compile(migraphx::make_target("ref"));

        auto last_hs = p.eval({}).back();
        std::vector<float> output_data;
        last_hs.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });

        std::vector<float> output_data_gold{
            -0.0327039, -0.0543852,  0.114378,   -0.0768855,  -0.0786602, -0.0613048, 0.179592,
            -0.071286,  -0.102509,   -0.0372696, 0.252296,    -0.144544,  -0.165194,  -0.0372928,
            0.273786,   -0.100877,   0.0319021,  -0.00298698, -0.0623361, 0.0598866,  0.074206,
            0.0124086,  -0.139544,   0.108016,   0.00496085,  0.0662588,  -0.048577,  -0.187329,
            -0.0458544, -0.0401315,  0.0737483,  -0.064505,   0.101585,   0.0687269,  -0.161725,
            -0.25617,   -0.00973633, -0.0552699, 0.0252681,   -0.0562072, 0.0855831,  -0.0171894,
            -0.140202,  0.0828391,   0.136898,   0.00160891,  -0.184812,  0.147774};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // forward, 8 args
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto pph  = mm->add_literal(migraphx::literal{pph_shape, pph_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);
        ic  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ic);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih,
            ic,
            pph);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}), hs);
        p.compile(migraphx::make_target("ref"));

        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.079753,  -0.289854,  0.160043,   0.115056,   0.186991,   -0.0624168, 0.205513,
            0.0836373, 0.0459033,  0.0414126,  0.272303,   0.0393149,  -0.058052,  0.0795391,
            0.266617,  -0.0128746, 0.294074,   -0.0319677, -0.0955337, 0.104168,   0.421857,
            0.0459771, -0.144955,  0.0720673,  0.218258,   0.0944405,  0.0431211,  -0.132394,
            0.0309878, 0.0971544,  0.149294,   -0.0492549, 0.022618,   -0.121195,  -0.4065,
            -0.252054, -0.0300906, -0.0890598, -0.135266,  -0.0413375, 0.103489,   0.0142918,
            -0.123408, 0.0401075,  0.187761,   0.0501726,  -0.121584,  0.0606723};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }

    // forward, last_output as program output, sequence length shorter
    // than max_seq_len
    {
        migraphx::program p;
        auto* mm      = p.get_main_module();
        auto seq_orig = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w        = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r        = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias     = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto ih       = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic       = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        migraphx::shape pad_seq_s{migraphx::shape::float_type, {batch_size, 2, input_size}};
        std::vector<float> pad_data(pad_seq_s.elements(), 0.0f);
        auto seq_p = mm->add_literal(migraphx::literal{pad_seq_s, pad_data});
        auto seq = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), seq_orig, seq_p);
        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data(batch_size, static_cast<int32_t>(seq_len));
        auto sql = mm->add_literal(seq_len_s, len_data);

        auto und = mm->add_instruction(migraphx::make_op("undefined"));

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);
        ic  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ic);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih,
            ic,
            und);
        auto last_output = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_output);
        p.compile(migraphx::make_target("ref"));

        auto last_hs = p.eval({}).back();
        std::vector<float> output_data;
        last_hs.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });

        std::vector<float> output_data_gold{-0.0847427,
                                            0.0874114,
                                            0.304256,
                                            -0.0585745,
                                            -0.0223018,
                                            0.131113,
                                            0.135643,
                                            -0.0566208,
                                            0.142701,
                                            0.0342236,
                                            -0.198664,
                                            0.0702607};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // seq_len = 1
    {
        seq_len = 1;
        migraphx::shape in_shape1{migraphx::shape::float_type, {batch_size, seq_len, input_size}};
        std::vector<float> input_data1{
            -0.5516, 0.2391, -1.6951, -0.4313, -0.9730, -0.2005, 2.3930, -0.5221, -0.1331};
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape1, input_data1});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto pph  = mm->add_literal(migraphx::literal{pph_shape, pph_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);
        ic  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ic);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih,
            ic,
            pph);
        auto last_output = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_output);
        p.compile(migraphx::make_target("ref"));

        auto hs_concat = p.eval({}).back();
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{0.079753,
                                        -0.289854,
                                        0.160043,
                                        0.115056,
                                        0.294074,
                                        -0.0319677,
                                        -0.0955337,
                                        0.104168,
                                        0.022618,
                                        -0.121195,
                                        -0.4065,
                                        -0.252054};
        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(lstm_reverse)
{
    std::size_t batch_size  = 3;
    std::size_t seq_len     = 4;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    std::vector<float> w_data{
        -0.2763, -0.4715, -0.3010, -0.2306, -0.2283, -0.2656, 0.2035,  0.3570,  -0.1499, 0.4390,
        -0.1843, 0.2351,  0.3357,  0.1217,  0.1401,  0.3300,  -0.0429, 0.3266,  0.4834,  -0.3914,
        -0.1480, 0.3734,  -0.0372, -0.1746, 0.0550,  0.4177,  -0.1332, 0.4391,  -0.3287, -0.4401,
        0.1486,  0.1346,  0.1048,  -0.4361, 0.0886,  -0.3840, -0.2730, -0.1710, 0.3274,  0.0169,
        -0.4462, 0.0729,  0.3983,  -0.0669, 0.0756,  0.4150,  -0.4684, -0.2522};

    std::vector<float> r_data{
        -0.4564, -0.4432, 0.1605,  0.4387,  0.0034,  0.4116,  0.2824,  0.4775,  -0.2729, -0.4707,
        0.1363,  0.2218,  0.0559,  0.2828,  0.2093,  0.4687,  0.3794,  -0.1069, -0.3049, 0.1430,
        -0.2506, 0.4644,  0.2755,  -0.3645, -0.3155, 0.1425,  0.2891,  0.1786,  -0.3274, 0.2365,
        0.2522,  -0.4312, -0.0562, -0.2748, 0.0776,  -0.3154, 0.2851,  -0.3930, -0.1174, 0.4360,
        0.2436,  0.0164,  -0.0680, 0.3403,  -0.2857, -0.0459, -0.2991, -0.2624, 0.4194,  -0.3291,
        -0.4659, 0.3300,  0.0454,  0.4981,  -0.4706, -0.4584, 0.2596,  0.2871,  -0.3509, -0.1910,
        0.3987,  -0.1687, -0.0032, -0.1038};

    std::vector<float> bias_data{-0.0258, 0.0073,  -0.4780, -0.4101, -0.3556, -0.1017, 0.3632,
                                 -0.1823, 0.1479,  0.1677,  -0.2603, 0.0381,  0.1575,  0.1896,
                                 0.4755,  -0.4794, 0.2167,  -0.4474, -0.3139, 0.1018,  0.4470,
                                 -0.4232, 0.3247,  -0.1636, -0.1582, -0.1703, 0.3920,  0.2055,
                                 -0.4386, 0.4208,  0.0717,  0.3789};

    std::vector<float> input_data{
        -0.5516, 0.2391, -1.6951, -0.4313, -0.9730, -0.2005, 2.3930,  -0.5221, -0.1331,
        -0.0910, 1.2122, -0.1952, 0.4661,  0.6494,  2.1332,  -1.0972, 0.9816,  0.1122,
        0.3577,  1.3508, -0.5366, 1.7449,  0.5483,  -0.0701, -0.4100, -2.2344, 0.3685,
        0.4583,  2.3794, 1.0372,  -0.8887, 0.7892,  -0.4012, -0.2818, -2.3374, 1.5310};

    std::vector<float> ih_data{1.5289,
                               1.0986,
                               0.6091,
                               1.6462,
                               0.8720,
                               0.5349,
                               -0.1962,
                               -1.7416,
                               -0.9912,
                               1.2831,
                               1.0896,
                               -0.6959};

    std::vector<float> ic_data{-0.8323,
                               0.3998,
                               0.1831,
                               0.5938,
                               2.7096,
                               -0.1790,
                               0.0022,
                               -0.8040,
                               0.1578,
                               0.0567,
                               0.8069,
                               -0.5141};

    std::vector<float> pph_data{-0.8271,
                                -0.5683,
                                0.4562,
                                -1.2545,
                                1.2729,
                                -0.4082,
                                -0.4392,
                                -0.9406,
                                0.7794,
                                1.8194,
                                -0.5811,
                                0.2166};

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape ic_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size}};
    float clip = 0.0f;
    // reverse, concatenation of hidden states as program output
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input_data});

        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto pph  = mm->add_literal(migraphx::literal{pph_shape, pph_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih,
            ic,
            pph);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{
            -0.120174, 0.043157,  0.117138,   -0.222188, 0.789732,   0.128538,   0.20909,
            0.0553812, -0.224905, 0.32421,    0.344048,  0.271694,   -0.175114,  -0.00543549,
            0.178681,  -0.266999, 0.928866,   0.113685,  0.220626,   -0.0432316, -0.063456,
            0.148524,  0.05108,   -0.0234895, -0.182201, -0.0232277, 0.235501,   -0.213485,
            0.960938,  0.133565,  0.269741,   0.130438,  -0.0252804, 0.267356,   0.146353,
            0.0789186, -0.185038, -0.026845,  0.177273,  -0.0774616, 0.946669,   0.0868676,
            0.044508,  -0.373961, -0.0681467, 0.382748,  0.230211,   -0.161537};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // reverse, sequence lengths are the same, but less than max_seq_lens
    {
        migraphx::program p;
        auto* mm      = p.get_main_module();
        auto seq_orig = mm->add_literal(migraphx::literal{in_shape, input_data});

        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto pph  = mm->add_literal(migraphx::literal{pph_shape, pph_data});

        migraphx::shape pad_seq_s{migraphx::shape::float_type, {2, batch_size, input_size}};
        std::vector<float> pad_data(pad_seq_s.elements(), 0.0f);
        auto seq_p = mm->add_literal(migraphx::literal{pad_seq_s, pad_data});
        auto seq = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), seq_orig, seq_p);
        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data(batch_size, static_cast<int32_t>(seq_len));
        auto sql = mm->add_literal(seq_len_s, len_data);
        mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih,
            ic,
            pph);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{
            -0.120174, 0.043157,  0.117138,   -0.222188, 0.789732,   0.128538,   0.20909,
            0.0553812, -0.224905, 0.32421,    0.344048,  0.271694,   -0.175114,  -0.00543549,
            0.178681,  -0.266999, 0.928866,   0.113685,  0.220626,   -0.0432316, -0.063456,
            0.148524,  0.05108,   -0.0234895, -0.182201, -0.0232277, 0.235501,   -0.213485,
            0.960938,  0.133565,  0.269741,   0.130438,  -0.0252804, 0.267356,   0.146353,
            0.0789186, -0.185038, -0.026845,  0.177273,  -0.0774616, 0.946669,   0.0868676,
            0.044508,  -0.373961, -0.0681467, 0.382748,  0.230211,   -0.161537,  0.0,
            0.0,       0.0,       0.0,        0.0,       0.0,        0.0,        0.0,
            0.0,       0.0,       0.0,        0.0,       0.0,        0.0,        0.0,
            0.0,       0.0,       0.0,        0.0,       0.0,        0.0,        0.0,
            0.0,       0.0};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // variable sequence lengths
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input_data});

        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto pph  = mm->add_literal(migraphx::literal{pph_shape, pph_data});

        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data{3, 2, 1};
        auto sql = mm->add_literal(seq_len_s, len_data);
        mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih,
            ic,
            pph);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{
            -0.126517, 0.0359124,  0.107453, -0.0617278, 0.911307,  0.11468,   0.114449,
            0.0196755, -0.102969,  0.295872, 0.515859,   0.246501,  -0.168327, 0.00023761,
            0.167567,  -0.0621982, 0.96657,  0.0755112,  0.0620917, -0.264845, 0,
            0,         0,          0,        -0.204545,  0.0146403, 0.210057,  0.0296268,
            0,         0,          0,        0,          0,         0,         0,
            0,         0,          0,        0,          0,         0,         0,
            0,         0,          0,        0,          0,         0};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // reverse, 3 args, last cell output as program output
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto hs  = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                  {"actv_func",
                   migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                       migraphx::make_op("tanh"),
                                                                       migraphx::make_op("tanh")})},
                  {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                  {"clip", clip},
                  {"input_forget", 0}}),
            seq,
            w,
            r);
        mm->add_instruction(migraphx::make_op("rnn_last_cell_output"), hs);

        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{-0.443077,
                                            -0.325425,
                                            -0.249367,
                                            -0.270812,
                                            0.122913,
                                            0.118537,
                                            0.0370199,
                                            -0.0164687,
                                            -0.00754759,
                                            0.141613,
                                            0.348002,
                                            0.667298};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // reverse, 3 args, 0 actv function
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto hs  = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                  {"actv_func", {}},
                  {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                  {"clip", clip},
                  {"input_forget", 0}}),
            seq,
            w,
            r);
        mm->add_instruction(migraphx::make_op("rnn_last_cell_output"), hs);

        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{-0.443077,
                                            -0.325425,
                                            -0.249367,
                                            -0.270812,
                                            0.122913,
                                            0.118537,
                                            0.0370199,
                                            -0.0164687,
                                            -0.00754759,
                                            0.141613,
                                            0.348002,
                                            0.667298};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }
}

TEST_CASE(lstm_reverse_layout)
{
    std::size_t batch_size  = 3;
    std::size_t seq_len     = 4;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    std::vector<float> w_data{
        -0.2763, -0.4715, -0.3010, -0.2306, -0.2283, -0.2656, 0.2035,  0.3570,  -0.1499, 0.4390,
        -0.1843, 0.2351,  0.3357,  0.1217,  0.1401,  0.3300,  -0.0429, 0.3266,  0.4834,  -0.3914,
        -0.1480, 0.3734,  -0.0372, -0.1746, 0.0550,  0.4177,  -0.1332, 0.4391,  -0.3287, -0.4401,
        0.1486,  0.1346,  0.1048,  -0.4361, 0.0886,  -0.3840, -0.2730, -0.1710, 0.3274,  0.0169,
        -0.4462, 0.0729,  0.3983,  -0.0669, 0.0756,  0.4150,  -0.4684, -0.2522};

    std::vector<float> r_data{
        -0.4564, -0.4432, 0.1605,  0.4387,  0.0034,  0.4116,  0.2824,  0.4775,  -0.2729, -0.4707,
        0.1363,  0.2218,  0.0559,  0.2828,  0.2093,  0.4687,  0.3794,  -0.1069, -0.3049, 0.1430,
        -0.2506, 0.4644,  0.2755,  -0.3645, -0.3155, 0.1425,  0.2891,  0.1786,  -0.3274, 0.2365,
        0.2522,  -0.4312, -0.0562, -0.2748, 0.0776,  -0.3154, 0.2851,  -0.3930, -0.1174, 0.4360,
        0.2436,  0.0164,  -0.0680, 0.3403,  -0.2857, -0.0459, -0.2991, -0.2624, 0.4194,  -0.3291,
        -0.4659, 0.3300,  0.0454,  0.4981,  -0.4706, -0.4584, 0.2596,  0.2871,  -0.3509, -0.1910,
        0.3987,  -0.1687, -0.0032, -0.1038};

    std::vector<float> bias_data{-0.0258, 0.0073,  -0.4780, -0.4101, -0.3556, -0.1017, 0.3632,
                                 -0.1823, 0.1479,  0.1677,  -0.2603, 0.0381,  0.1575,  0.1896,
                                 0.4755,  -0.4794, 0.2167,  -0.4474, -0.3139, 0.1018,  0.4470,
                                 -0.4232, 0.3247,  -0.1636, -0.1582, -0.1703, 0.3920,  0.2055,
                                 -0.4386, 0.4208,  0.0717,  0.3789};

    std::vector<float> input_data{
        -0.5516, 0.2391, -1.6951, -0.0910, 1.2122,  -0.1952, 0.3577,  1.3508,  -0.5366,
        0.4583,  2.3794, 1.0372,  -0.4313, -0.9730, -0.2005, 0.4661,  0.6494,  2.1332,
        1.7449,  0.5483, -0.0701, -0.8887, 0.7892,  -0.4012, 2.3930,  -0.5221, -0.1331,
        -1.0972, 0.9816, 0.1122,  -0.4100, -2.2344, 0.3685,  -0.2818, -2.3374, 1.5310};

    std::vector<float> ih_data{1.5289,
                               1.0986,
                               0.6091,
                               1.6462,
                               0.8720,
                               0.5349,
                               -0.1962,
                               -1.7416,
                               -0.9912,
                               1.2831,
                               1.0896,
                               -0.6959};

    std::vector<float> ic_data{-0.8323,
                               0.3998,
                               0.1831,
                               0.5938,
                               2.7096,
                               -0.1790,
                               0.0022,
                               -0.8040,
                               0.1578,
                               0.0567,
                               0.8069,
                               -0.5141};

    std::vector<float> pph_data{-0.8271,
                                -0.5683,
                                0.4562,
                                -1.2545,
                                1.2729,
                                -0.4082,
                                -0.4392,
                                -0.9406,
                                0.7794,
                                1.8194,
                                -0.5811,
                                0.2166};

    migraphx::shape in_shape{migraphx::shape::float_type, {batch_size, seq_len, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {batch_size, num_dirct, hidden_size}};
    migraphx::shape ic_shape{migraphx::shape::float_type, {batch_size, num_dirct, hidden_size}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size}};
    float clip = 0.0f;

    // reverse, concatenation of hidden states as program output
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input_data});

        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto pph  = mm->add_literal(migraphx::literal{pph_shape, pph_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);
        ic  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ic);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih,
            ic,
            pph);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}), hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{
            -0.120174, 0.043157,   0.117138,   -0.222188, -0.175114,  -0.00543549, 0.178681,
            -0.266999, -0.182201,  -0.0232277, 0.235501,  -0.213485,  -0.185038,   -0.026845,
            0.177273,  -0.0774616, 0.789732,   0.128538,  0.20909,    0.0553812,   0.928866,
            0.113685,  0.220626,   -0.0432316, 0.960938,  0.133565,   0.269741,    0.130438,
            0.946669,  0.0868676,  0.044508,   -0.373961, -0.224905,  0.32421,     0.344048,
            0.271694,  -0.063456,  0.148524,   0.05108,   -0.0234895, -0.0252804,  0.267356,
            0.146353,  0.0789186,  -0.0681467, 0.382748,  0.230211,   -0.161537};

        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // reverse, sequence lengths are the same, but less than max_seq_lens
    {
        migraphx::program p;
        auto* mm      = p.get_main_module();
        auto seq_orig = mm->add_literal(migraphx::literal{in_shape, input_data});

        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto pph  = mm->add_literal(migraphx::literal{pph_shape, pph_data});

        migraphx::shape pad_seq_s{migraphx::shape::float_type, {batch_size, 2, input_size}};
        std::vector<float> pad_data(pad_seq_s.elements(), 0.0f);
        auto seq_p = mm->add_literal(migraphx::literal{pad_seq_s, pad_data});
        auto seq = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), seq_orig, seq_p);
        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data(batch_size, static_cast<int32_t>(seq_len));
        auto sql = mm->add_literal(seq_len_s, len_data);

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);
        ic  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ic);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih,
            ic,
            pph);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}), hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });

        std::vector<float> output_data_gold{
            -0.120174,  0.043157,   0.117138,   -0.222188,  -0.175114,  -0.00543549, 0.178681,
            -0.266999,  -0.182201,  -0.0232277, 0.235501,   -0.213485,  -0.185038,   -0.026845,
            0.177273,   -0.0774616, 0.0,        0.0,        0.0,        0.0,         0.0,
            0.0,        0.0,        0.0,        0.789732,   0.128538,   0.20909,     0.0553812,
            0.928866,   0.113685,   0.220626,   -0.0432316, 0.960938,   0.133565,    0.269741,
            0.130438,   0.946669,   0.0868676,  0.044508,   -0.373961,  0.0,         0.0,
            0.0,        0.0,        0.0,        0.0,        0.0,        0.0,         -0.224905,
            0.32421,    0.344048,   0.271694,   -0.063456,  0.148524,   0.05108,     -0.0234895,
            -0.0252804, 0.267356,   0.146353,   0.0789186,  -0.0681467, 0.382748,    0.230211,
            -0.161537,  0.0,        0.0,        0.0,        0.0,        0.0,         0.0,
            0.0,        0.0};

        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // variable sequence lengths
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input_data});

        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto pph  = mm->add_literal(migraphx::literal{pph_shape, pph_data});

        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data{3, 2, 1};
        auto sql = mm->add_literal(seq_len_s, len_data);

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);
        ic  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ic);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih,
            ic,
            pph);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}), hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });

        std::vector<float> output_data_gold{
            -0.126517,  0.0359124, 0.107453,  -0.0617278, -0.168327, 0.00023761, 0.167567,
            -0.0621982, -0.204545, 0.0146403, 0.210057,   0.0296268, 0,          0,
            0,          0,         0.911307,  0.11468,    0.114449,  0.0196755,  0.96657,
            0.0755112,  0.0620917, -0.264845, 0,          0,         0,          0,
            0,          0,         0,         0,          -0.102969, 0.295872,   0.515859,
            0.246501,   0,         0,         0,          0,         0,          0,
            0,          0,         0,         0,          0,         0};

        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // reverse, 3 args, last cell output as program output
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);

        auto hs  = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                  {"actv_func",
                   migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                       migraphx::make_op("tanh"),
                                                                       migraphx::make_op("tanh")})},
                  {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                  {"clip", clip},
                  {"input_forget", 0}}),
            seq,
            w,
            r);
        auto cell_output = mm->add_instruction(migraphx::make_op("rnn_last_cell_output"), hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), cell_output);

        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{-0.443077,
                                            -0.325425,
                                            -0.249367,
                                            -0.270812,
                                            0.122913,
                                            0.118537,
                                            0.0370199,
                                            -0.0164687,
                                            -0.00754759,
                                            0.141613,
                                            0.348002,
                                            0.667298};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }
}

// lstm activation function test
TEST_CASE(lstm_reverse_actv)
{
    std::size_t batch_size  = 3;
    std::size_t seq_len     = 4;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    std::vector<float> w_data{
        -0.2763, -0.4715, -0.3010, -0.2306, -0.2283, -0.2656, 0.2035,  0.3570,  -0.1499, 0.4390,
        -0.1843, 0.2351,  0.3357,  0.1217,  0.1401,  0.3300,  -0.0429, 0.3266,  0.4834,  -0.3914,
        -0.1480, 0.3734,  -0.0372, -0.1746, 0.0550,  0.4177,  -0.1332, 0.4391,  -0.3287, -0.4401,
        0.1486,  0.1346,  0.1048,  -0.4361, 0.0886,  -0.3840, -0.2730, -0.1710, 0.3274,  0.0169,
        -0.4462, 0.0729,  0.3983,  -0.0669, 0.0756,  0.4150,  -0.4684, -0.2522};

    std::vector<float> r_data{
        -0.4564, -0.4432, 0.1605,  0.4387,  0.0034,  0.4116,  0.2824,  0.4775,  -0.2729, -0.4707,
        0.1363,  0.2218,  0.0559,  0.2828,  0.2093,  0.4687,  0.3794,  -0.1069, -0.3049, 0.1430,
        -0.2506, 0.4644,  0.2755,  -0.3645, -0.3155, 0.1425,  0.2891,  0.1786,  -0.3274, 0.2365,
        0.2522,  -0.4312, -0.0562, -0.2748, 0.0776,  -0.3154, 0.2851,  -0.3930, -0.1174, 0.4360,
        0.2436,  0.0164,  -0.0680, 0.3403,  -0.2857, -0.0459, -0.2991, -0.2624, 0.4194,  -0.3291,
        -0.4659, 0.3300,  0.0454,  0.4981,  -0.4706, -0.4584, 0.2596,  0.2871,  -0.3509, -0.1910,
        0.3987,  -0.1687, -0.0032, -0.1038};

    std::vector<float> bias_data{-0.0258, 0.0073,  -0.4780, -0.4101, -0.3556, -0.1017, 0.3632,
                                 -0.1823, 0.1479,  0.1677,  -0.2603, 0.0381,  0.1575,  0.1896,
                                 0.4755,  -0.4794, 0.2167,  -0.4474, -0.3139, 0.1018,  0.4470,
                                 -0.4232, 0.3247,  -0.1636, -0.1582, -0.1703, 0.3920,  0.2055,
                                 -0.4386, 0.4208,  0.0717,  0.3789};

    std::vector<float> input_data{
        -0.5516, 0.2391, -1.6951, -0.4313, -0.9730, -0.2005, 2.3930,  -0.5221, -0.1331,
        -0.0910, 1.2122, -0.1952, 0.4661,  0.6494,  2.1332,  -1.0972, 0.9816,  0.1122,
        0.3577,  1.3508, -0.5366, 1.7449,  0.5483,  -0.0701, -0.4100, -2.2344, 0.3685,
        0.4583,  2.3794, 1.0372,  -0.8887, 0.7892,  -0.4012, -0.2818, -2.3374, 1.5310};

    std::vector<float> ih_data{1.5289,
                               1.0986,
                               0.6091,
                               1.6462,
                               0.8720,
                               0.5349,
                               -0.1962,
                               -1.7416,
                               -0.9912,
                               1.2831,
                               1.0896,
                               -0.6959};

    std::vector<float> ic_data{-0.8323,
                               0.3998,
                               0.1831,
                               0.5938,
                               2.7096,
                               -0.1790,
                               0.0022,
                               -0.8040,
                               0.1578,
                               0.0567,
                               0.8069,
                               -0.5141};

    std::vector<float> pph_data{-0.8271,
                                -0.5683,
                                0.4562,
                                -1.2545,
                                1.2729,
                                -0.4082,
                                -0.4392,
                                -0.9406,
                                0.7794,
                                1.8194,
                                -0.5811,
                                0.2166};

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape ic_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size}};
    float clip = 0.0f;
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input_data});

        auto w = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r = mm->add_literal(migraphx::literal{r_shape, r_data});
        mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(
                      std::vector<migraphx::operation>{migraphx::make_op("sigmoid")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{
            0.246078, 0.199709, 0.303753, 0.301178, 0.264634, 0.304661, 0.349371, 0.288934,
            0.405483, 0.445586, 0.515814, 0.473186, 0.301937, 0.264893, 0.254353, 0.269231,
            0.359258, 0.400097, 0.288884, 0.247329, 0.276519, 0.264249, 0.1769,   0.23213,
            0.310306, 0.262902, 0.276964, 0.295002, 0.373802, 0.366785, 0.419791, 0.393216,
            0.262827, 0.371441, 0.369022, 0.298262, 0.334143, 0.309444, 0.174822, 0.251634,
            0.244564, 0.214386, 0.185994, 0.226699, 0.28445,  0.376092, 0.338326, 0.259502};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // reverse, 3 args, 2 actv functions
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input_data});

        auto w  = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r  = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{
                      migraphx::make_op("tanh"), migraphx::make_op("sigmoid")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{-0.132123,
                                            -0.37531,
                                            -0.12943,
                                            -0.00798307,
                                            -0.133882,
                                            -0.0251383,
                                            0.0486486,
                                            -0.0220606,
                                            0.292495,
                                            0.233866,
                                            0.48646,
                                            0.481844};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // reverse, 3 args, seq_len = 1, concatenation of hidden states as program output
    {
        seq_len = 1;
        std::vector<float> input_data1{
            -0.5516, 0.2391, -1.6951, -0.4313, -0.9730, -0.2005, 2.3930, -0.5221, -0.1331};
        migraphx::shape in_shape1{migraphx::shape::float_type, {seq_len, batch_size, input_size}};

        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape1, input_data1});

        auto w = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r = mm->add_literal(migraphx::literal{r_shape, r_data});
        mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{-0.104351,
                                            -0.0471426,
                                            -0.0905753,
                                            0.01506,
                                            0.059797,
                                            0.104239,
                                            -0.0266768,
                                            0.0727547,
                                            -0.146298,
                                            0.070535,
                                            0.327809,
                                            0.407388};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }
}

TEST_CASE(lstm_bidirectional)
{
    std::size_t batch_size  = 3;
    std::size_t seq_len     = 4;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 2;
    std::vector<float> w_data{
        0.1236,  -0.3942, 0.4149,  0.0795,  0.4934,  -0.2858, 0.2602,  -0.3098, 0.0567,  0.3344,
        0.3607,  -0.0551, 0.4952,  0.3799,  0.0630,  -0.3532, 0.0023,  -0.0592, 0.4267,  0.2382,
        -0.0784, -0.0032, -0.2476, -0.0206, -0.4963, 0.4837,  0.0827,  0.0123,  -0.1203, -0.0279,
        -0.0049, 0.4721,  -0.3564, -0.1286, 0.4090,  -0.0504, 0.0575,  -0.2138, 0.1071,  0.1976,
        -0.0758, 0.0139,  -0.0761, 0.3991,  -0.2965, -0.4845, -0.1496, 0.3285,  -0.2763, -0.4715,
        -0.3010, -0.2306, -0.2283, -0.2656, 0.2035,  0.3570,  -0.1499, 0.4390,  -0.1843, 0.2351,
        0.3357,  0.1217,  0.1401,  0.3300,  -0.0429, 0.3266,  0.4834,  -0.3914, -0.1480, 0.3734,
        -0.0372, -0.1746, 0.0550,  0.4177,  -0.1332, 0.4391,  -0.3287, -0.4401, 0.1486,  0.1346,
        0.1048,  -0.4361, 0.0886,  -0.3840, -0.2730, -0.1710, 0.3274,  0.0169,  -0.4462, 0.0729,
        0.3983,  -0.0669, 0.0756,  0.4150,  -0.4684, -0.2522};

    std::vector<float> r_data{
        0.1237,  0.1229,  -0.0766, -0.1144, -0.1186, 0.2922,  0.2478,  0.3159,  -0.0522, 0.1685,
        -0.4621, 0.1728,  0.0670,  -0.2458, -0.3835, -0.4589, -0.3109, 0.4908,  -0.0133, -0.1858,
        -0.0590, -0.0347, -0.2353, -0.0671, -0.3812, -0.0004, -0.1432, 0.2406,  0.1033,  -0.0265,
        -0.3902, 0.0755,  0.3733,  0.4383,  -0.3140, 0.2537,  -0.1818, -0.4127, 0.3506,  0.2562,
        0.2926,  0.1620,  -0.4849, -0.4861, 0.4426,  0.2106,  -0.0005, 0.4418,  -0.2926, -0.3100,
        0.1500,  -0.0362, -0.3801, -0.0065, -0.0631, 0.1277,  0.2315,  0.4087,  -0.3963, -0.4161,
        -0.2169, -0.1344, 0.3468,  -0.2260, -0.4564, -0.4432, 0.1605,  0.4387,  0.0034,  0.4116,
        0.2824,  0.4775,  -0.2729, -0.4707, 0.1363,  0.2218,  0.0559,  0.2828,  0.2093,  0.4687,
        0.3794,  -0.1069, -0.3049, 0.1430,  -0.2506, 0.4644,  0.2755,  -0.3645, -0.3155, 0.1425,
        0.2891,  0.1786,  -0.3274, 0.2365,  0.2522,  -0.4312, -0.0562, -0.2748, 0.0776,  -0.3154,
        0.2851,  -0.3930, -0.1174, 0.4360,  0.2436,  0.0164,  -0.0680, 0.3403,  -0.2857, -0.0459,
        -0.2991, -0.2624, 0.4194,  -0.3291, -0.4659, 0.3300,  0.0454,  0.4981,  -0.4706, -0.4584,
        0.2596,  0.2871,  -0.3509, -0.1910, 0.3987,  -0.1687, -0.0032, -0.1038};

    std::vector<float> bias_data{
        0.0088,  0.1183,  0.1642,  -0.2631, -0.1330, -0.4008, 0.3881,  -0.4407, -0.2760, 0.1274,
        -0.0083, -0.2885, 0.3949,  -0.0182, 0.4445,  0.3477,  0.2266,  0.3423,  -0.0674, -0.4067,
        0.0807,  0.1109,  -0.2036, 0.1782,  -0.2467, -0.0730, -0.4216, 0.0316,  -0.3025, 0.3637,
        -0.3181, -0.4655, -0.0258, 0.0073,  -0.4780, -0.4101, -0.3556, -0.1017, 0.3632,  -0.1823,
        0.1479,  0.1677,  -0.2603, 0.0381,  0.1575,  0.1896,  0.4755,  -0.4794, 0.2167,  -0.4474,
        -0.3139, 0.1018,  0.4470,  -0.4232, 0.3247,  -0.1636, -0.1582, -0.1703, 0.3920,  0.2055,
        -0.4386, 0.4208,  0.0717,  0.3789};

    std::vector<float> input_data{
        -0.5516, 0.2391, -1.6951, -0.4313, -0.9730, -0.2005, 2.3930,  -0.5221, -0.1331,
        -0.0910, 1.2122, -0.1952, 0.4661,  0.6494,  2.1332,  -1.0972, 0.9816,  0.1122,
        0.3577,  1.3508, -0.5366, 1.7449,  0.5483,  -0.0701, -0.4100, -2.2344, 0.3685,
        0.4583,  2.3794, 1.0372,  -0.8887, 0.7892,  -0.4012, -0.2818, -2.3374, 1.5310};

    std::vector<float> ih_data{1.9104,  -1.9004, 0.3337,  0.5741, 0.5671, 0.0458,
                               0.4514,  -0.8968, -0.9201, 0.1962, 0.5771, -0.5332,
                               1.5289,  1.0986,  0.6091,  1.6462, 0.8720, 0.5349,
                               -0.1962, -1.7416, -0.9912, 1.2831, 1.0896, -0.6959};

    std::vector<float> ic_data{0.9569,  -0.5981, 1.1312,  1.0945,  1.1055,  -0.1212,
                               -0.9097, 0.7831,  -1.6991, -1.9498, -1.2567, -0.4114,
                               -0.8323, 0.3998,  0.1831,  0.5938,  2.7096,  -0.1790,
                               0.0022,  -0.8040, 0.1578,  0.0567,  0.8069,  -0.5141};

    std::vector<float> pph_data{1.84369764,  0.68413646, -0.44892886, -1.50904413, 0.3860796,
                                -0.52186625, 1.08474445, -1.80867321, 1.32594529,  0.4336262,
                                -0.83699064, 0.49162736, -0.8271,     -0.5683,     0.4562,
                                -1.2545,     1.2729,     -0.4082,     -0.4392,     -0.9406,
                                0.7794,      1.8194,     -0.5811,     0.2166};
    float clip = 0.0f;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape ic_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size}};

    // concatenation of hidden states as program output
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto pph  = mm->add_literal(migraphx::literal{pph_shape, pph_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih,
            ic,
            pph);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{
            0.079753,   -0.289854,  0.160043,    0.115056,   0.294074,   -0.0319677, -0.0955337,
            0.104168,   0.022618,   -0.121195,   -0.4065,    -0.252054,  -0.120174,  0.043157,
            0.117138,   -0.222188,  0.789732,    0.128538,   0.20909,    0.0553812,  -0.224905,
            0.32421,    0.344048,   0.271694,    0.186991,   -0.0624168, 0.205513,   0.0836373,
            0.421857,   0.0459771,  -0.144955,   0.0720673,  -0.0300906, -0.0890598, -0.135266,
            -0.0413375, -0.175114,  -0.00543549, 0.178681,   -0.266999,  0.928866,   0.113685,
            0.220626,   -0.0432316, -0.063456,   0.148524,   0.05108,    -0.0234895, 0.0459032,
            0.0414126,  0.272303,   0.0393149,   0.218258,   0.0944405,  0.0431211,  -0.132394,
            0.103489,   0.0142918,  -0.123408,   0.0401075,  -0.182201,  -0.0232277, 0.235501,
            -0.213485,  0.960938,   0.133565,    0.269741,   0.130438,   -0.0252804, 0.267356,
            0.146353,   0.0789186,  -0.058052,   0.0795391,  0.266617,   -0.0128746, 0.0309878,
            0.0971544,  0.149294,   -0.0492549,  0.187761,   0.0501726,  -0.121584,  0.0606723,
            -0.185038,  -0.026845,  0.177273,    -0.0774616, 0.946669,   0.0868676,  0.044508,
            -0.373961,  -0.0681467, 0.382748,    0.230211,   -0.161537};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // last hidden state as program output
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto pph  = mm->add_literal(migraphx::literal{pph_shape, pph_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        auto hs   = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                   {"actv_func",
                    migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                        migraphx::make_op("tanh"),
                                                                        migraphx::make_op("tanh"),
                                                                        migraphx::make_op("sigmoid"),
                                                                        migraphx::make_op("tanh"),
                                                                        migraphx::make_op("tanh")})},
                   {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                   {"clip", clip},
                   {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih,
            ic,
            pph);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{
            -0.058052, 0.0795391, 0.266617,  -0.0128746, 0.0309878, 0.0971544, 0.149294, -0.0492549,
            0.187761,  0.0501726, -0.121584, 0.0606723,  -0.120174, 0.043157,  0.117138, -0.222188,
            0.789732,  0.128538,  0.20909,   0.0553812,  -0.224905, 0.32421,   0.344048, 0.271694};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // last cell output as program output
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto pph  = mm->add_literal(migraphx::literal{pph_shape, pph_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));
        auto hs   = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                   {"actv_func",
                    migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                        migraphx::make_op("tanh"),
                                                                        migraphx::make_op("tanh"),
                                                                        migraphx::make_op("sigmoid"),
                                                                        migraphx::make_op("tanh"),
                                                                        migraphx::make_op("tanh")})},
                   {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                   {"clip", clip},
                   {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih,
            ic,
            pph);
        mm->add_instruction(migraphx::make_op("rnn_last_cell_output"), hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{
            -0.077353, 0.245616, 0.361023,  -0.0443759, 0.0685243, 0.20465,  0.277867, -0.112934,
            0.67312,   0.120508, -0.726968, 0.113845,   -0.889294, 0.182463, 0.186512, -0.402334,
            1.48161,   0.524116, 0.347113,  0.181813,   -0.434265, 0.747833, 0.416053, 0.558713};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // 3 args, concatenation of hidden states as program output
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});
        mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{
            -0.0327039, -0.0543852, 0.114378,  -0.0768855, 0.0319021,   -0.00298698, -0.0623361,
            0.0598866,  0.101585,   0.0687269, -0.161725,  -0.25617,    -0.162851,   -0.102647,
            -0.113827,  -0.142818,  0.0513685, 0.0547876,  0.0201981,   -0.00808453, -0.00520328,
            0.0945081,  0.264123,   0.410805,  -0.0786602, -0.0613048,  0.179592,    -0.071286,
            0.074206,   0.0124086,  -0.139544, 0.108016,   -0.00973633, -0.0552699,  0.0252681,
            -0.0562072, -0.123496,  -0.153616, -0.032874,  -0.195349,   0.0192675,   -0.108636,
            0.098927,   -0.140733,  0.162602,  0.0143099,  -0.0455534,  0.0151574,   -0.102509,
            -0.0372696, 0.252296,   -0.144544, 0.00496085, 0.0662588,   -0.048577,   -0.187329,
            0.0855831,  -0.0171894, -0.140202, 0.0828391,  -0.1073,     -0.150145,   0.015065,
            -0.192699,  -0.112764,  -0.120496, 0.155754,   0.148256,    0.208491,    0.348432,
            0.0291103,  0.230275,   -0.165194, -0.0372928, 0.273786,    -0.100877,   -0.0458544,
            -0.0401315, 0.0737483,  -0.064505, 0.136898,   0.00160891,  -0.184812,   0.147774,
            -0.021205,  -0.125423,  0.0206439, -0.187097,  -0.0051453,  -0.0767618,  -0.0735348,
            -0.0826436, 0.214159,   0.262295,  0.0247127,  0.14472};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // sequence length is 1, contenation of hidden state as program output
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        seq_len  = 1;
        migraphx::shape in_shape1{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        std::vector<float> input_data1{
            -0.5516, 0.2391, -1.6951, -0.4313, -0.9730, -0.2005, 2.3930, -0.5221, -0.1331};
        auto seq = mm->add_literal(migraphx::literal{in_shape1, input_data1});
        auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});
        mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{
            -0.0327039, -0.0543852, 0.114378,   -0.0768855, 0.0319021, -0.00298698,
            -0.0623361, 0.0598866,  0.101585,   0.0687269,  -0.161725, -0.25617,
            -0.104351,  -0.0471426, -0.0905753, 0.01506,    0.059797,  0.104239,
            -0.0266768, 0.0727547,  -0.146298,  0.070535,   0.327809,  0.407388};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }
}

TEST_CASE(lstm_bidirectional_layout)
{
    std::size_t batch_size  = 3;
    std::size_t seq_len     = 4;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 2;
    std::vector<float> w_data{
        0.1236,  -0.3942, 0.4149,  0.0795,  0.4934,  -0.2858, 0.2602,  -0.3098, 0.0567,  0.3344,
        0.3607,  -0.0551, 0.4952,  0.3799,  0.0630,  -0.3532, 0.0023,  -0.0592, 0.4267,  0.2382,
        -0.0784, -0.0032, -0.2476, -0.0206, -0.4963, 0.4837,  0.0827,  0.0123,  -0.1203, -0.0279,
        -0.0049, 0.4721,  -0.3564, -0.1286, 0.4090,  -0.0504, 0.0575,  -0.2138, 0.1071,  0.1976,
        -0.0758, 0.0139,  -0.0761, 0.3991,  -0.2965, -0.4845, -0.1496, 0.3285,  -0.2763, -0.4715,
        -0.3010, -0.2306, -0.2283, -0.2656, 0.2035,  0.3570,  -0.1499, 0.4390,  -0.1843, 0.2351,
        0.3357,  0.1217,  0.1401,  0.3300,  -0.0429, 0.3266,  0.4834,  -0.3914, -0.1480, 0.3734,
        -0.0372, -0.1746, 0.0550,  0.4177,  -0.1332, 0.4391,  -0.3287, -0.4401, 0.1486,  0.1346,
        0.1048,  -0.4361, 0.0886,  -0.3840, -0.2730, -0.1710, 0.3274,  0.0169,  -0.4462, 0.0729,
        0.3983,  -0.0669, 0.0756,  0.4150,  -0.4684, -0.2522};

    std::vector<float> r_data{
        0.1237,  0.1229,  -0.0766, -0.1144, -0.1186, 0.2922,  0.2478,  0.3159,  -0.0522, 0.1685,
        -0.4621, 0.1728,  0.0670,  -0.2458, -0.3835, -0.4589, -0.3109, 0.4908,  -0.0133, -0.1858,
        -0.0590, -0.0347, -0.2353, -0.0671, -0.3812, -0.0004, -0.1432, 0.2406,  0.1033,  -0.0265,
        -0.3902, 0.0755,  0.3733,  0.4383,  -0.3140, 0.2537,  -0.1818, -0.4127, 0.3506,  0.2562,
        0.2926,  0.1620,  -0.4849, -0.4861, 0.4426,  0.2106,  -0.0005, 0.4418,  -0.2926, -0.3100,
        0.1500,  -0.0362, -0.3801, -0.0065, -0.0631, 0.1277,  0.2315,  0.4087,  -0.3963, -0.4161,
        -0.2169, -0.1344, 0.3468,  -0.2260, -0.4564, -0.4432, 0.1605,  0.4387,  0.0034,  0.4116,
        0.2824,  0.4775,  -0.2729, -0.4707, 0.1363,  0.2218,  0.0559,  0.2828,  0.2093,  0.4687,
        0.3794,  -0.1069, -0.3049, 0.1430,  -0.2506, 0.4644,  0.2755,  -0.3645, -0.3155, 0.1425,
        0.2891,  0.1786,  -0.3274, 0.2365,  0.2522,  -0.4312, -0.0562, -0.2748, 0.0776,  -0.3154,
        0.2851,  -0.3930, -0.1174, 0.4360,  0.2436,  0.0164,  -0.0680, 0.3403,  -0.2857, -0.0459,
        -0.2991, -0.2624, 0.4194,  -0.3291, -0.4659, 0.3300,  0.0454,  0.4981,  -0.4706, -0.4584,
        0.2596,  0.2871,  -0.3509, -0.1910, 0.3987,  -0.1687, -0.0032, -0.1038};

    std::vector<float> bias_data{
        0.0088,  0.1183,  0.1642,  -0.2631, -0.1330, -0.4008, 0.3881,  -0.4407, -0.2760, 0.1274,
        -0.0083, -0.2885, 0.3949,  -0.0182, 0.4445,  0.3477,  0.2266,  0.3423,  -0.0674, -0.4067,
        0.0807,  0.1109,  -0.2036, 0.1782,  -0.2467, -0.0730, -0.4216, 0.0316,  -0.3025, 0.3637,
        -0.3181, -0.4655, -0.0258, 0.0073,  -0.4780, -0.4101, -0.3556, -0.1017, 0.3632,  -0.1823,
        0.1479,  0.1677,  -0.2603, 0.0381,  0.1575,  0.1896,  0.4755,  -0.4794, 0.2167,  -0.4474,
        -0.3139, 0.1018,  0.4470,  -0.4232, 0.3247,  -0.1636, -0.1582, -0.1703, 0.3920,  0.2055,
        -0.4386, 0.4208,  0.0717,  0.3789};

    std::vector<float> input_data{
        -0.5516, 0.2391, -1.6951, -0.0910, 1.2122,  -0.1952, 0.3577,  1.3508,  -0.5366,
        0.4583,  2.3794, 1.0372,  -0.4313, -0.9730, -0.2005, 0.4661,  0.6494,  2.1332,
        1.7449,  0.5483, -0.0701, -0.8887, 0.7892,  -0.4012, 2.3930,  -0.5221, -0.1331,
        -1.0972, 0.9816, 0.1122,  -0.4100, -2.2344, 0.3685,  -0.2818, -2.3374, 1.5310};

    std::vector<float> ih_data{1.9104, -1.9004, 0.3337,  0.5741,  1.5289,  1.0986,
                               0.6091, 1.6462,  0.5671,  0.0458,  0.4514,  -0.8968,
                               0.8720, 0.5349,  -0.1962, -1.7416, -0.9201, 0.1962,
                               0.5771, -0.5332, -0.9912, 1.2831,  1.0896,  -0.6959};

    std::vector<float> ic_data{0.9569,  -0.5981, 1.1312, 1.0945,  -0.8323, 0.3998,
                               0.1831,  0.5938,  1.1055, -0.1212, -0.9097, 0.7831,
                               2.7096,  -0.1790, 0.0022, -0.8040, -1.6991, -1.9498,
                               -1.2567, -0.4114, 0.1578, 0.0567,  0.8069,  -0.5141};

    std::vector<float> pph_data{1.84369764,  0.68413646, -0.44892886, -1.50904413, 0.3860796,
                                -0.52186625, 1.08474445, -1.80867321, 1.32594529,  0.4336262,
                                -0.83699064, 0.49162736, -0.8271,     -0.5683,     0.4562,
                                -1.2545,     1.2729,     -0.4082,     -0.4392,     -0.9406,
                                0.7794,      1.8194,     -0.5811,     0.2166};
    float clip = 0.0f;
    migraphx::shape in_shape{migraphx::shape::float_type, {batch_size, seq_len, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {batch_size, num_dirct, hidden_size}};
    migraphx::shape ic_shape{migraphx::shape::float_type, {batch_size, num_dirct, hidden_size}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size}};

    // concatenation of hidden states as program output
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto pph  = mm->add_literal(migraphx::literal{pph_shape, pph_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);
        ic  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ic);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih,
            ic,
            pph);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}), hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });

        std::vector<float> output_data_gold{
            0.079753,   -0.289854,  0.160043,   0.115056,   -0.120174, 0.043157,   0.117138,
            -0.222188,  0.186991,   -0.0624168, 0.205513,   0.0836373, -0.175114,  -0.00543549,
            0.178681,   -0.266999,  0.0459032,  0.0414126,  0.272303,  0.0393149,  -0.182201,
            -0.0232277, 0.235501,   -0.213485,  -0.058052,  0.0795391, 0.266617,   -0.0128746,
            -0.185038,  -0.026845,  0.177273,   -0.0774616, 0.294074,  -0.0319677, -0.0955337,
            0.104168,   0.789732,   0.128538,   0.20909,    0.0553812, 0.421857,   0.0459771,
            -0.144955,  0.0720673,  0.928866,   0.113685,   0.220626,  -0.0432316, 0.218258,
            0.0944405,  0.0431211,  -0.132394,  0.960938,   0.133565,  0.269741,   0.130438,
            0.0309878,  0.0971544,  0.149294,   -0.0492549, 0.946669,  0.0868676,  0.044508,
            -0.373961,  0.022618,   -0.121195,  -0.4065,    -0.252054, -0.224905,  0.32421,
            0.344048,   0.271694,   -0.0300906, -0.0890598, -0.135266, -0.0413375, -0.063456,
            0.148524,   0.05108,    -0.0234895, 0.103489,   0.0142918, -0.123408,  0.0401075,
            -0.0252804, 0.267356,   0.146353,   0.0789186,  0.187761,  0.0501726,  -0.121584,
            0.0606723,  -0.0681467, 0.382748,   0.230211,   -0.161537};

        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // last hidden state as program output
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto pph  = mm->add_literal(migraphx::literal{pph_shape, pph_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);
        ic  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ic);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih,
            ic,
            pph);
        auto last_output = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_output);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{
            -0.058052, 0.0795391, 0.266617,  -0.0128746, -0.120174, 0.043157, 0.117138, -0.222188,
            0.0309878, 0.0971544, 0.149294,  -0.0492549, 0.789732,  0.128538, 0.20909,  0.0553812,
            0.187761,  0.0501726, -0.121584, 0.0606723,  -0.224905, 0.32421,  0.344048, 0.271694};

        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // last cell output as program output
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto pph  = mm->add_literal(migraphx::literal{pph_shape, pph_data});
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);
        ic  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ic);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            und,
            ih,
            ic,
            pph);
        auto cell_output = mm->add_instruction(migraphx::make_op("rnn_last_cell_output"), hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), cell_output);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{
            -0.077353, 0.245616, 0.361023,  -0.0443759, -0.889294, 0.182463, 0.186512, -0.402334,
            0.0685243, 0.20465,  0.277867,  -0.112934,  1.48161,   0.524116, 0.347113, 0.181813,
            0.67312,   0.120508, -0.726968, 0.113845,   -0.434265, 0.747833, 0.416053, 0.558713};

        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // 3 args, concatenation of hidden states as program output
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}), hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });

        std::vector<float> output_data_gold{
            -0.0327039, -0.0543852, 0.114378,    -0.0768855, -0.162851,   -0.102647,   -0.113827,
            -0.142818,  -0.0786602, -0.0613048,  0.179592,   -0.071286,   -0.123496,   -0.153616,
            -0.032874,  -0.195349,  -0.102509,   -0.0372696, 0.252296,    -0.144544,   -0.1073,
            -0.150145,  0.015065,   -0.192699,   -0.165194,  -0.0372928,  0.273786,    -0.100877,
            -0.021205,  -0.125423,  0.0206439,   -0.187097,  0.0319021,   -0.00298698, -0.0623361,
            0.0598866,  0.0513685,  0.0547876,   0.0201981,  -0.00808453, 0.074206,    0.0124086,
            -0.139544,  0.108016,   0.0192675,   -0.108636,  0.098927,    -0.140733,   0.00496085,
            0.0662588,  -0.048577,  -0.187329,   -0.112764,  -0.120496,   0.155754,    0.148256,
            -0.0458544, -0.0401315, 0.0737483,   -0.064505,  -0.0051453,  -0.0767618,  -0.0735348,
            -0.0826436, 0.101585,   0.0687269,   -0.161725,  -0.25617,    -0.00520328, 0.0945081,
            0.264123,   0.410805,   -0.00973633, -0.0552699, 0.0252681,   -0.0562072,  0.162602,
            0.0143099,  -0.0455534, 0.0151574,   0.0855831,  -0.0171894,  -0.140202,   0.0828391,
            0.208491,   0.348432,   0.0291103,   0.230275,   0.136898,    0.00160891,  -0.184812,
            0.147774,   0.214159,   0.262295,    0.0247127,  0.14472};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // sequence length is 1, contenation of hidden state as program output
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        seq_len  = 1;
        migraphx::shape in_shape1{migraphx::shape::float_type, {batch_size, seq_len, input_size}};
        std::vector<float> input_data1{
            -0.5516, 0.2391, -1.6951, -0.4313, -0.9730, -0.2005, 2.3930, -0.5221, -0.1331};
        auto seq = mm->add_literal(migraphx::literal{in_shape1, input_data1});
        auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}), hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });

        std::vector<float> output_data_gold{
            -0.0327039, -0.0543852, 0.114378,   -0.0768855,  -0.104351,  -0.0471426,
            -0.0905753, 0.01506,    0.0319021,  -0.00298698, -0.0623361, 0.0598866,
            0.059797,   0.104239,   -0.0266768, 0.0727547,   0.101585,   0.0687269,
            -0.161725,  -0.25617,   -0.146298,  0.070535,    0.327809,   0.407388};

        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }
}

TEST_CASE(lstm_bidirectional_var_seq_lens)
{
    std::size_t batch_size  = 3;
    std::size_t seq_len     = 4;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 2;
    std::vector<float> w_data{
        0.1236,  -0.3942, 0.4149,  0.0795,  0.4934,  -0.2858, 0.2602,  -0.3098, 0.0567,  0.3344,
        0.3607,  -0.0551, 0.4952,  0.3799,  0.0630,  -0.3532, 0.0023,  -0.0592, 0.4267,  0.2382,
        -0.0784, -0.0032, -0.2476, -0.0206, -0.4963, 0.4837,  0.0827,  0.0123,  -0.1203, -0.0279,
        -0.0049, 0.4721,  -0.3564, -0.1286, 0.4090,  -0.0504, 0.0575,  -0.2138, 0.1071,  0.1976,
        -0.0758, 0.0139,  -0.0761, 0.3991,  -0.2965, -0.4845, -0.1496, 0.3285,  -0.2763, -0.4715,
        -0.3010, -0.2306, -0.2283, -0.2656, 0.2035,  0.3570,  -0.1499, 0.4390,  -0.1843, 0.2351,
        0.3357,  0.1217,  0.1401,  0.3300,  -0.0429, 0.3266,  0.4834,  -0.3914, -0.1480, 0.3734,
        -0.0372, -0.1746, 0.0550,  0.4177,  -0.1332, 0.4391,  -0.3287, -0.4401, 0.1486,  0.1346,
        0.1048,  -0.4361, 0.0886,  -0.3840, -0.2730, -0.1710, 0.3274,  0.0169,  -0.4462, 0.0729,
        0.3983,  -0.0669, 0.0756,  0.4150,  -0.4684, -0.2522};

    std::vector<float> r_data{
        0.1237,  0.1229,  -0.0766, -0.1144, -0.1186, 0.2922,  0.2478,  0.3159,  -0.0522, 0.1685,
        -0.4621, 0.1728,  0.0670,  -0.2458, -0.3835, -0.4589, -0.3109, 0.4908,  -0.0133, -0.1858,
        -0.0590, -0.0347, -0.2353, -0.0671, -0.3812, -0.0004, -0.1432, 0.2406,  0.1033,  -0.0265,
        -0.3902, 0.0755,  0.3733,  0.4383,  -0.3140, 0.2537,  -0.1818, -0.4127, 0.3506,  0.2562,
        0.2926,  0.1620,  -0.4849, -0.4861, 0.4426,  0.2106,  -0.0005, 0.4418,  -0.2926, -0.3100,
        0.1500,  -0.0362, -0.3801, -0.0065, -0.0631, 0.1277,  0.2315,  0.4087,  -0.3963, -0.4161,
        -0.2169, -0.1344, 0.3468,  -0.2260, -0.4564, -0.4432, 0.1605,  0.4387,  0.0034,  0.4116,
        0.2824,  0.4775,  -0.2729, -0.4707, 0.1363,  0.2218,  0.0559,  0.2828,  0.2093,  0.4687,
        0.3794,  -0.1069, -0.3049, 0.1430,  -0.2506, 0.4644,  0.2755,  -0.3645, -0.3155, 0.1425,
        0.2891,  0.1786,  -0.3274, 0.2365,  0.2522,  -0.4312, -0.0562, -0.2748, 0.0776,  -0.3154,
        0.2851,  -0.3930, -0.1174, 0.4360,  0.2436,  0.0164,  -0.0680, 0.3403,  -0.2857, -0.0459,
        -0.2991, -0.2624, 0.4194,  -0.3291, -0.4659, 0.3300,  0.0454,  0.4981,  -0.4706, -0.4584,
        0.2596,  0.2871,  -0.3509, -0.1910, 0.3987,  -0.1687, -0.0032, -0.1038};

    std::vector<float> bias_data{
        0.0088,  0.1183,  0.1642,  -0.2631, -0.1330, -0.4008, 0.3881,  -0.4407, -0.2760, 0.1274,
        -0.0083, -0.2885, 0.3949,  -0.0182, 0.4445,  0.3477,  0.2266,  0.3423,  -0.0674, -0.4067,
        0.0807,  0.1109,  -0.2036, 0.1782,  -0.2467, -0.0730, -0.4216, 0.0316,  -0.3025, 0.3637,
        -0.3181, -0.4655, -0.0258, 0.0073,  -0.4780, -0.4101, -0.3556, -0.1017, 0.3632,  -0.1823,
        0.1479,  0.1677,  -0.2603, 0.0381,  0.1575,  0.1896,  0.4755,  -0.4794, 0.2167,  -0.4474,
        -0.3139, 0.1018,  0.4470,  -0.4232, 0.3247,  -0.1636, -0.1582, -0.1703, 0.3920,  0.2055,
        -0.4386, 0.4208,  0.0717,  0.3789};

    std::vector<float> input_data{
        -0.5516, 0.2391, -1.6951, -0.4313, -0.9730, -0.2005, 2.3930,  -0.5221, -0.1331,
        -0.0910, 1.2122, -0.1952, 0.4661,  0.6494,  2.1332,  -1.0972, 0.9816,  0.1122,
        0.3577,  1.3508, -0.5366, 1.7449,  0.5483,  -0.0701, -0.4100, -2.2344, 0.3685,
        0.4583,  2.3794, 1.0372,  -0.8887, 0.7892,  -0.4012, -0.2818, -2.3374, 1.5310};

    std::vector<float> ih_data{1.9104,  -1.9004, 0.3337,  0.5741, 0.5671, 0.0458,
                               0.4514,  -0.8968, -0.9201, 0.1962, 0.5771, -0.5332,
                               1.5289,  1.0986,  0.6091,  1.6462, 0.8720, 0.5349,
                               -0.1962, -1.7416, -0.9912, 1.2831, 1.0896, -0.6959};

    std::vector<float> ic_data{0.9569,  -0.5981, 1.1312,  1.0945,  1.1055,  -0.1212,
                               -0.9097, 0.7831,  -1.6991, -1.9498, -1.2567, -0.4114,
                               -0.8323, 0.3998,  0.1831,  0.5938,  2.7096,  -0.1790,
                               0.0022,  -0.8040, 0.1578,  0.0567,  0.8069,  -0.5141};

    std::vector<float> pph_data{1.84369764,  0.68413646, -0.44892886, -1.50904413, 0.3860796,
                                -0.52186625, 1.08474445, -1.80867321, 1.32594529,  0.4336262,
                                -0.83699064, 0.49162736, -0.8271,     -0.5683,     0.4562,
                                -1.2545,     1.2729,     -0.4082,     -0.4392,     -0.9406,
                                0.7794,      1.8194,     -0.5811,     0.2166};

    float clip = 0.0f;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape ic_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size}};

    // concatenation of hidden states as program output
    {
        std::vector<int> sl_data{1, 2, 3};
        migraphx::shape sl_shape{migraphx::shape::int32_type, {batch_size}};
        migraphx::program p;
        auto* mm    = p.get_main_module();
        auto seq    = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto ih     = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic     = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto w      = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r      = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias   = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto pph    = mm->add_literal(migraphx::literal{pph_shape, pph_data});
        auto sql    = mm->add_literal(migraphx::literal{sl_shape, sl_data});
        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih,
            ic,
            pph);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto lco = mm->add_instruction(migraphx::make_op("rnn_last_cell_output"), out_hs);
        mm->add_return({out_hs, lho, lco});
        p.compile(migraphx::make_target("ref"));

        auto outputs = p.eval({});
        const auto& arg_hs  = outputs.front();
        const auto& arg_lho = outputs.at(1);
        const auto& arg_lco = outputs.at(2);

        std::vector<float> output_data;
        std::vector<float> last_output_data;
        std::vector<float> last_cell_data;

        arg_hs.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        arg_lho.visit([&](auto output) { last_output_data.assign(output.begin(), output.end()); });
        arg_lco.visit([&](auto output) { last_cell_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{
            0.079753,   -0.289854, 0.160043,  0.115056,  0.294074,   -0.0319677, -0.0955337,
            0.104168,   0.022618,  -0.121195, -0.4065,   -0.252054,  -0.141643,  0.0451978,
            0.140804,   0.0745128, 0.911307,  0.11468,   0.114449,   0.0196755,  -0.262807,
            0.275286,   0.358395,  0.266267,  0,         0,          0,          0,
            0.421857,   0.0459771, -0.144955, 0.0720673, -0.0300906, -0.0890598, -0.135266,
            -0.0413375, 0,         0,         0,         0,          0.96657,    0.0755112,
            0.0620917,  -0.264845, -0.128254, 0.125398,  0.0665142,  -0.163651,  0,
            0,          0,         0,         0,         0,          0,          0,
            0.103489,   0.0142918, -0.123408, 0.0401075, 0,          0,          0,
            0,          0,         0,         0,         0,          -0.0644683, 0.371512,
            0.212431,   -0.116131, 0,         0,         0,          0,          0,
            0,          0,         0,         0,         0,          0,          0,
            0,          0,         0,         0,         0,          0,          0,
            0,          0,         0,         0,         0};
        std::vector<float> last_output_data_gold{
            0.079753, -0.289854, 0.160043,  0.115056,  0.421857,  0.0459771, -0.144955, 0.0720673,
            0.103489, 0.0142918, -0.123408, 0.0401075, -0.141643, 0.0451978, 0.140804,  0.0745128,
            0.911307, 0.11468,   0.114449,  0.0196755, -0.262807, 0.275286,  0.358395,  0.266267};
        std::vector<float> last_cell_data_gold{
            0.600582, -0.601197, 0.353558,  0.789097,  0.737121,  0.134902, -0.303595, 0.241948,
            0.391174, 0.0308845, -0.561745, 0.0730323, -0.326822, 0.301121, 0.219523,  0.415242,
            2.08242,  0.442513,  0.187127,  0.0577626, -0.611307, 0.55454,  0.4364,    0.509436};

        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(last_output_data, last_output_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(last_cell_data, last_cell_data_gold));
    }

    // last cell output as program output
    {
        migraphx::program p;
        auto* mm      = p.get_main_module();
        auto seq_orig = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto ih       = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic       = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto w        = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r        = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias     = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto pph      = mm->add_literal(migraphx::literal{pph_shape, pph_data});
        migraphx::shape pad_seq_s{migraphx::shape::float_type, {2, batch_size, input_size}};
        std::vector<float> pad_data(pad_seq_s.elements(), 0.0f);
        auto seq_p = mm->add_literal(migraphx::literal{pad_seq_s, pad_data});
        auto seq = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), seq_orig, seq_p);
        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data(batch_size, static_cast<int32_t>(seq_len));
        auto sql = mm->add_literal(seq_len_s, len_data);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih,
            ic,
            pph);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        auto lco = mm->add_instruction(migraphx::make_op("rnn_last_cell_output"), hs);
        mm->add_return({hs, lho, lco});
        p.compile(migraphx::make_target("ref"));
        auto outputs = p.eval({});
        const auto& res_hs  = outputs.at(0);
        const auto& res_lho = outputs.at(1);
        const auto& res_lco = outputs.at(2);
        std::vector<float> hs_data;
        std::vector<float> lho_data;
        std::vector<float> lco_data;
        res_hs.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });
        res_lho.visit([&](auto output) { lho_data.assign(output.begin(), output.end()); });
        res_lco.visit([&](auto output) { lco_data.assign(output.begin(), output.end()); });
        std::vector<float> hs_data_gold{
            0.079753,   -0.289854,  0.160043,    0.115056,   0.294074,   -0.0319677, -0.0955337,
            0.104168,   0.022618,   -0.121195,   -0.4065,    -0.252054,  -0.120174,  0.043157,
            0.117138,   -0.222188,  0.789732,    0.128538,   0.20909,    0.0553812,  -0.224905,
            0.32421,    0.344048,   0.271694,    0.186991,   -0.0624168, 0.205513,   0.0836373,
            0.421857,   0.0459771,  -0.144955,   0.0720673,  -0.0300906, -0.0890598, -0.135266,
            -0.0413375, -0.175114,  -0.00543549, 0.178681,   -0.266999,  0.928866,   0.113685,
            0.220626,   -0.0432316, -0.063456,   0.148524,   0.05108,    -0.0234895, 0.0459033,
            0.0414126,  0.272303,   0.0393149,   0.218258,   0.0944405,  0.0431211,  -0.132394,
            0.103489,   0.0142918,  -0.123408,   0.0401075,  -0.182201,  -0.0232277, 0.235501,
            -0.213485,  0.960938,   0.133565,    0.269741,   0.130438,   -0.0252804, 0.267356,
            0.146353,   0.0789186,  -0.058052,   0.0795391,  0.266617,   -0.0128746, 0.0309878,
            0.0971544,  0.149294,   -0.0492549,  0.187761,   0.0501726,  -0.121584,  0.0606723,
            -0.185038,  -0.026845,  0.177273,    -0.0774616, 0.946669,   0.0868676,  0.044508,
            -0.373961,  -0.0681467, 0.382748,    0.230211,   -0.161537,  0,          0,
            0,          0,          0,           0,          0,          0,          0,
            0,          0,          0,           0,          0,          0,          0,
            0,          0,          0,           0,          0,          0,          0,
            0,          0,          0,           0,          0,          0,          0,
            0,          0,          0,           0,          0,          0,          0,
            0,          0,          0,           0,          0,          0,          0,
            0,          0,          0,           0};
        std::vector<float> lho_data_gold{
            -0.058052, 0.0795391, 0.266617,  -0.0128746, 0.0309878, 0.0971544, 0.149294, -0.0492549,
            0.187761,  0.0501726, -0.121584, 0.0606723,  -0.120174, 0.043157,  0.117138, -0.222188,
            0.789732,  0.128538,  0.20909,   0.0553812,  -0.224905, 0.32421,   0.344048, 0.271694};
        std::vector<float> lco_data_gold{
            -0.077353, 0.245616, 0.361023,  -0.0443759, 0.0685243, 0.20465,  0.277867, -0.112934,
            0.67312,   0.120508, -0.726968, 0.113845,   -0.889294, 0.182463, 0.186512, -0.402334,
            1.48161,   0.524116, 0.347113,  0.181813,   -0.434265, 0.747833, 0.416053, 0.558713};
        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(lho_data, lho_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(lco_data, lco_data_gold));
    }
}

TEST_CASE(lstm_bidirectional_var_seq_lens_layout)
{
    std::size_t batch_size  = 3;
    std::size_t seq_len     = 4;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 2;
    std::vector<float> w_data{
        0.1236,  -0.3942, 0.4149,  0.0795,  0.4934,  -0.2858, 0.2602,  -0.3098, 0.0567,  0.3344,
        0.3607,  -0.0551, 0.4952,  0.3799,  0.0630,  -0.3532, 0.0023,  -0.0592, 0.4267,  0.2382,
        -0.0784, -0.0032, -0.2476, -0.0206, -0.4963, 0.4837,  0.0827,  0.0123,  -0.1203, -0.0279,
        -0.0049, 0.4721,  -0.3564, -0.1286, 0.4090,  -0.0504, 0.0575,  -0.2138, 0.1071,  0.1976,
        -0.0758, 0.0139,  -0.0761, 0.3991,  -0.2965, -0.4845, -0.1496, 0.3285,  -0.2763, -0.4715,
        -0.3010, -0.2306, -0.2283, -0.2656, 0.2035,  0.3570,  -0.1499, 0.4390,  -0.1843, 0.2351,
        0.3357,  0.1217,  0.1401,  0.3300,  -0.0429, 0.3266,  0.4834,  -0.3914, -0.1480, 0.3734,
        -0.0372, -0.1746, 0.0550,  0.4177,  -0.1332, 0.4391,  -0.3287, -0.4401, 0.1486,  0.1346,
        0.1048,  -0.4361, 0.0886,  -0.3840, -0.2730, -0.1710, 0.3274,  0.0169,  -0.4462, 0.0729,
        0.3983,  -0.0669, 0.0756,  0.4150,  -0.4684, -0.2522};

    std::vector<float> r_data{
        0.1237,  0.1229,  -0.0766, -0.1144, -0.1186, 0.2922,  0.2478,  0.3159,  -0.0522, 0.1685,
        -0.4621, 0.1728,  0.0670,  -0.2458, -0.3835, -0.4589, -0.3109, 0.4908,  -0.0133, -0.1858,
        -0.0590, -0.0347, -0.2353, -0.0671, -0.3812, -0.0004, -0.1432, 0.2406,  0.1033,  -0.0265,
        -0.3902, 0.0755,  0.3733,  0.4383,  -0.3140, 0.2537,  -0.1818, -0.4127, 0.3506,  0.2562,
        0.2926,  0.1620,  -0.4849, -0.4861, 0.4426,  0.2106,  -0.0005, 0.4418,  -0.2926, -0.3100,
        0.1500,  -0.0362, -0.3801, -0.0065, -0.0631, 0.1277,  0.2315,  0.4087,  -0.3963, -0.4161,
        -0.2169, -0.1344, 0.3468,  -0.2260, -0.4564, -0.4432, 0.1605,  0.4387,  0.0034,  0.4116,
        0.2824,  0.4775,  -0.2729, -0.4707, 0.1363,  0.2218,  0.0559,  0.2828,  0.2093,  0.4687,
        0.3794,  -0.1069, -0.3049, 0.1430,  -0.2506, 0.4644,  0.2755,  -0.3645, -0.3155, 0.1425,
        0.2891,  0.1786,  -0.3274, 0.2365,  0.2522,  -0.4312, -0.0562, -0.2748, 0.0776,  -0.3154,
        0.2851,  -0.3930, -0.1174, 0.4360,  0.2436,  0.0164,  -0.0680, 0.3403,  -0.2857, -0.0459,
        -0.2991, -0.2624, 0.4194,  -0.3291, -0.4659, 0.3300,  0.0454,  0.4981,  -0.4706, -0.4584,
        0.2596,  0.2871,  -0.3509, -0.1910, 0.3987,  -0.1687, -0.0032, -0.1038};

    std::vector<float> bias_data{
        0.0088,  0.1183,  0.1642,  -0.2631, -0.1330, -0.4008, 0.3881,  -0.4407, -0.2760, 0.1274,
        -0.0083, -0.2885, 0.3949,  -0.0182, 0.4445,  0.3477,  0.2266,  0.3423,  -0.0674, -0.4067,
        0.0807,  0.1109,  -0.2036, 0.1782,  -0.2467, -0.0730, -0.4216, 0.0316,  -0.3025, 0.3637,
        -0.3181, -0.4655, -0.0258, 0.0073,  -0.4780, -0.4101, -0.3556, -0.1017, 0.3632,  -0.1823,
        0.1479,  0.1677,  -0.2603, 0.0381,  0.1575,  0.1896,  0.4755,  -0.4794, 0.2167,  -0.4474,
        -0.3139, 0.1018,  0.4470,  -0.4232, 0.3247,  -0.1636, -0.1582, -0.1703, 0.3920,  0.2055,
        -0.4386, 0.4208,  0.0717,  0.3789};

    std::vector<float> input_data{
        -0.5516, 0.2391, -1.6951, -0.0910, 1.2122,  -0.1952, 0.3577,  1.3508,  -0.5366,
        0.4583,  2.3794, 1.0372,  -0.4313, -0.9730, -0.2005, 0.4661,  0.6494,  2.1332,
        1.7449,  0.5483, -0.0701, -0.8887, 0.7892,  -0.4012, 2.3930,  -0.5221, -0.1331,
        -1.0972, 0.9816, 0.1122,  -0.4100, -2.2344, 0.3685,  -0.2818, -2.3374, 1.5310};

    std::vector<float> ih_data{1.9104, -1.9004, 0.3337,  0.5741,  1.5289,  1.0986,
                               0.6091, 1.6462,  0.5671,  0.0458,  0.4514,  -0.8968,
                               0.8720, 0.5349,  -0.1962, -1.7416, -0.9201, 0.1962,
                               0.5771, -0.5332, -0.9912, 1.2831,  1.0896,  -0.6959};

    std::vector<float> ic_data{0.9569,  -0.5981, 1.1312, 1.0945,  -0.8323, 0.3998,
                               0.1831,  0.5938,  1.1055, -0.1212, -0.9097, 0.7831,
                               2.7096,  -0.1790, 0.0022, -0.8040, -1.6991, -1.9498,
                               -1.2567, -0.4114, 0.1578, 0.0567,  0.8069,  -0.5141};

    std::vector<float> pph_data{1.84369764,  0.68413646, -0.44892886, -1.50904413, 0.3860796,
                                -0.52186625, 1.08474445, -1.80867321, 1.32594529,  0.4336262,
                                -0.83699064, 0.49162736, -0.8271,     -0.5683,     0.4562,
                                -1.2545,     1.2729,     -0.4082,     -0.4392,     -0.9406,
                                0.7794,      1.8194,     -0.5811,     0.2166};

    float clip = 0.0f;
    migraphx::shape in_shape{migraphx::shape::float_type, {batch_size, seq_len, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {batch_size, num_dirct, hidden_size}};
    migraphx::shape ic_shape{migraphx::shape::float_type, {batch_size, num_dirct, hidden_size}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size}};

    // concatenation of hidden states as program output
    {
        std::vector<int> sl_data{1, 2, 3};
        migraphx::shape sl_shape{migraphx::shape::int32_type, {batch_size}};
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto pph  = mm->add_literal(migraphx::literal{pph_shape, pph_data});
        auto sql  = mm->add_literal(migraphx::literal{sl_shape, sl_data});

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);
        ic  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ic);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih,
            ic,
            pph);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto lco = mm->add_instruction(migraphx::make_op("rnn_last_cell_output"), out_hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        lho    = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), lho);
        lco    = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), lco);
        mm->add_return({out_hs, lho, lco});
        p.compile(migraphx::make_target("ref"));

        auto outputs = p.eval({});
        const auto& arg_hs  = outputs.front();
        const auto& arg_lho = outputs.at(1);
        const auto& arg_lco = outputs.at(2);

        std::vector<float> output_data;
        std::vector<float> last_output_data;
        std::vector<float> last_cell_data;

        arg_hs.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        arg_lho.visit([&](auto output) { last_output_data.assign(output.begin(), output.end()); });
        arg_lco.visit([&](auto output) { last_cell_data.assign(output.begin(), output.end()); });

        std::vector<float> output_data_gold{
            0.079753,   -0.289854, 0.160043,   0.115056,   -0.141643, 0.0451978,  0.140804,
            0.0745128,  0,         0,          0,          0,         0,          0,
            0,          0,         0,          0,          0,         0,          0,
            0,          0,         0,          0,          0,         0,          0,
            0,          0,         0,          0,          0.294074,  -0.0319677, -0.0955337,
            0.104168,   0.911307,  0.11468,    0.114449,   0.0196755, 0.421857,   0.0459771,
            -0.144955,  0.0720673, 0.96657,    0.0755112,  0.0620917, -0.264845,  0,
            0,          0,         0,          0,          0,         0,          0,
            0,          0,         0,          0,          0,         0,          0,
            0,          0.022618,  -0.121195,  -0.4065,    -0.252054, -0.262807,  0.275286,
            0.358395,   0.266267,  -0.0300906, -0.0890598, -0.135266, -0.0413375, -0.128254,
            0.125398,   0.0665142, -0.163651,  0.103489,   0.0142918, -0.123408,  0.0401075,
            -0.0644683, 0.371512,  0.212431,   -0.116131,  0,         0,          0,
            0,          0,         0,          0,          0};

        std::vector<float> last_output_data_gold{
            0.079753, -0.289854, 0.160043,  0.115056,  -0.141643, 0.0451978, 0.140804, 0.0745128,
            0.421857, 0.0459771, -0.144955, 0.0720673, 0.911307,  0.11468,   0.114449, 0.0196755,
            0.103489, 0.0142918, -0.123408, 0.0401075, -0.262807, 0.275286,  0.358395, 0.266267};

        std::vector<float> last_cell_data_gold{
            0.600582, -0.601197, 0.353558,  0.789097,  -0.326822, 0.301121, 0.219523, 0.415242,
            0.737121, 0.134902,  -0.303595, 0.241948,  2.08242,   0.442513, 0.187127, 0.0577626,
            0.391174, 0.0308845, -0.561745, 0.0730323, -0.611307, 0.55454,  0.4364,   0.509436};

        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(last_output_data, last_output_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(last_cell_data, last_cell_data_gold));
    }

    // last cell output as program output
    {
        migraphx::program p;
        auto* mm      = p.get_main_module();
        auto seq_orig = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto ih       = mm->add_literal(migraphx::literal{ih_shape, ih_data});
        auto ic       = mm->add_literal(migraphx::literal{ic_shape, ic_data});
        auto w        = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r        = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto bias     = mm->add_literal(migraphx::literal{b_shape, bias_data});
        auto pph      = mm->add_literal(migraphx::literal{pph_shape, pph_data});
        migraphx::shape pad_seq_s{migraphx::shape::float_type, {batch_size, 2, input_size}};
        std::vector<float> pad_data(pad_seq_s.elements(), 0.0f);
        auto seq_p = mm->add_literal(migraphx::literal{pad_seq_s, pad_data});
        auto seq = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), seq_orig, seq_p);
        migraphx::shape seq_len_s{migraphx::shape::int32_type, {batch_size}};
        std::vector<int32_t> len_data(batch_size, static_cast<int32_t>(seq_len));
        auto sql = mm->add_literal(seq_len_s, len_data);

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);
        ic  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ic);

        auto hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r,
            bias,
            sql,
            ih,
            ic,
            pph);
        auto lho = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        auto lco = mm->add_instruction(migraphx::make_op("rnn_last_cell_output"), hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        hs  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}), hs);
        lho = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), lho);
        lco = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), lco);
        mm->add_return({hs, lho, lco});
        p.compile(migraphx::make_target("ref"));

        auto outputs = p.eval({});
        const auto& res_hs  = outputs.at(0);
        const auto& res_lho = outputs.at(1);
        const auto& res_lco = outputs.at(2);
        std::vector<float> hs_data;
        std::vector<float> lho_data;
        std::vector<float> lco_data;
        res_hs.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });
        res_lho.visit([&](auto output) { lho_data.assign(output.begin(), output.end()); });
        res_lco.visit([&](auto output) { lco_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.079753,   -0.289854,  0.160043,   0.115056,   -0.120174,  0.043157,   0.117138,
            -0.222188,  0.186991,   -0.0624168, 0.205513,   0.0836373,  -0.175114,  -0.00543549,
            0.178681,   -0.266999,  0.0459033,  0.0414126,  0.272303,   0.0393149,  -0.182201,
            -0.0232277, 0.235501,   -0.213485,  -0.058052,  0.0795391,  0.266617,   -0.0128746,
            -0.185038,  -0.026845,  0.177273,   -0.0774616, 0,          0,          0,
            0,          0,          0,          0,          0,          0,          0,
            0,          0,          0,          0,          0,          0,          0.294074,
            -0.0319677, -0.0955337, 0.104168,   0.789732,   0.128538,   0.20909,    0.0553812,
            0.421857,   0.0459771,  -0.144955,  0.0720673,  0.928866,   0.113685,   0.220626,
            -0.0432316, 0.218258,   0.0944405,  0.0431211,  -0.132394,  0.960938,   0.133565,
            0.269741,   0.130438,   0.0309878,  0.0971544,  0.149294,   -0.0492549, 0.946669,
            0.0868676,  0.044508,   -0.373961,  0,          0,          0,          0,
            0,          0,          0,          0,          0,          0,          0,
            0,          0,          0,          0,          0,          0.022618,   -0.121195,
            -0.4065,    -0.252054,  -0.224905,  0.32421,    0.344048,   0.271694,   -0.0300906,
            -0.0890598, -0.135266,  -0.0413375, -0.063456,  0.148524,   0.05108,    -0.0234895,
            0.103489,   0.0142918,  -0.123408,  0.0401075,  -0.0252804, 0.267356,   0.146353,
            0.0789186,  0.187761,   0.0501726,  -0.121584,  0.0606723,  -0.0681467, 0.382748,
            0.230211,   -0.161537,  0,          0,          0,          0,          0,
            0,          0,          0,          0,          0,          0,          0,
            0,          0,          0,          0};

        std::vector<float> lho_data_gold{
            -0.058052, 0.0795391, 0.266617,  -0.0128746, -0.120174, 0.043157, 0.117138, -0.222188,
            0.0309878, 0.0971544, 0.149294,  -0.0492549, 0.789732,  0.128538, 0.20909,  0.0553812,
            0.187761,  0.0501726, -0.121584, 0.0606723,  -0.224905, 0.32421,  0.344048, 0.271694};

        std::vector<float> lco_data_gold{
            -0.077353, 0.245616, 0.361023,  -0.0443759, -0.889294, 0.182463, 0.186512, -0.402334,
            0.0685243, 0.20465,  0.277867,  -0.112934,  1.48161,   0.524116, 0.347113, 0.181813,
            0.67312,   0.120508, -0.726968, 0.113845,   -0.434265, 0.747833, 0.416053, 0.558713};

        EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(lho_data, lho_data_gold));
        EXPECT(migraphx::verify::verify_rms_range(lco_data, lco_data_gold));
    }
}

TEST_CASE(lstm_bidirectional_actv_func)
{
    std::size_t batch_size  = 3;
    std::size_t seq_len     = 4;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 2;
    std::vector<float> w_data{
        0.1236,  -0.3942, 0.4149,  0.0795,  0.4934,  -0.2858, 0.2602,  -0.3098, 0.0567,  0.3344,
        0.3607,  -0.0551, 0.4952,  0.3799,  0.0630,  -0.3532, 0.0023,  -0.0592, 0.4267,  0.2382,
        -0.0784, -0.0032, -0.2476, -0.0206, -0.4963, 0.4837,  0.0827,  0.0123,  -0.1203, -0.0279,
        -0.0049, 0.4721,  -0.3564, -0.1286, 0.4090,  -0.0504, 0.0575,  -0.2138, 0.1071,  0.1976,
        -0.0758, 0.0139,  -0.0761, 0.3991,  -0.2965, -0.4845, -0.1496, 0.3285,  -0.2763, -0.4715,
        -0.3010, -0.2306, -0.2283, -0.2656, 0.2035,  0.3570,  -0.1499, 0.4390,  -0.1843, 0.2351,
        0.3357,  0.1217,  0.1401,  0.3300,  -0.0429, 0.3266,  0.4834,  -0.3914, -0.1480, 0.3734,
        -0.0372, -0.1746, 0.0550,  0.4177,  -0.1332, 0.4391,  -0.3287, -0.4401, 0.1486,  0.1346,
        0.1048,  -0.4361, 0.0886,  -0.3840, -0.2730, -0.1710, 0.3274,  0.0169,  -0.4462, 0.0729,
        0.3983,  -0.0669, 0.0756,  0.4150,  -0.4684, -0.2522};

    std::vector<float> r_data{
        0.1237,  0.1229,  -0.0766, -0.1144, -0.1186, 0.2922,  0.2478,  0.3159,  -0.0522, 0.1685,
        -0.4621, 0.1728,  0.0670,  -0.2458, -0.3835, -0.4589, -0.3109, 0.4908,  -0.0133, -0.1858,
        -0.0590, -0.0347, -0.2353, -0.0671, -0.3812, -0.0004, -0.1432, 0.2406,  0.1033,  -0.0265,
        -0.3902, 0.0755,  0.3733,  0.4383,  -0.3140, 0.2537,  -0.1818, -0.4127, 0.3506,  0.2562,
        0.2926,  0.1620,  -0.4849, -0.4861, 0.4426,  0.2106,  -0.0005, 0.4418,  -0.2926, -0.3100,
        0.1500,  -0.0362, -0.3801, -0.0065, -0.0631, 0.1277,  0.2315,  0.4087,  -0.3963, -0.4161,
        -0.2169, -0.1344, 0.3468,  -0.2260, -0.4564, -0.4432, 0.1605,  0.4387,  0.0034,  0.4116,
        0.2824,  0.4775,  -0.2729, -0.4707, 0.1363,  0.2218,  0.0559,  0.2828,  0.2093,  0.4687,
        0.3794,  -0.1069, -0.3049, 0.1430,  -0.2506, 0.4644,  0.2755,  -0.3645, -0.3155, 0.1425,
        0.2891,  0.1786,  -0.3274, 0.2365,  0.2522,  -0.4312, -0.0562, -0.2748, 0.0776,  -0.3154,
        0.2851,  -0.3930, -0.1174, 0.4360,  0.2436,  0.0164,  -0.0680, 0.3403,  -0.2857, -0.0459,
        -0.2991, -0.2624, 0.4194,  -0.3291, -0.4659, 0.3300,  0.0454,  0.4981,  -0.4706, -0.4584,
        0.2596,  0.2871,  -0.3509, -0.1910, 0.3987,  -0.1687, -0.0032, -0.1038};

    std::vector<float> input_data{
        -0.5516, 0.2391, -1.6951, -0.4313, -0.9730, -0.2005, 2.3930,  -0.5221, -0.1331,
        -0.0910, 1.2122, -0.1952, 0.4661,  0.6494,  2.1332,  -1.0972, 0.9816,  0.1122,
        0.3577,  1.3508, -0.5366, 1.7449,  0.5483,  -0.0701, -0.4100, -2.2344, 0.3685,
        0.4583,  2.3794, 1.0372,  -0.8887, 0.7892,  -0.4012, -0.2818, -2.3374, 1.5310};

    float clip = 0.0f;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, hidden_size}};
    // 3 args, 0 actv func
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});
        mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func", {}},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{
            -0.0327039, -0.0543852, 0.114378,  -0.0768855, 0.0319021,   -0.00298698, -0.0623361,
            0.0598866,  0.101585,   0.0687269, -0.161725,  -0.25617,    -0.162851,   -0.102647,
            -0.113827,  -0.142818,  0.0513685, 0.0547876,  0.0201981,   -0.00808453, -0.00520328,
            0.0945081,  0.264123,   0.410805,  -0.0786602, -0.0613048,  0.179592,    -0.071286,
            0.074206,   0.0124086,  -0.139544, 0.108016,   -0.00973633, -0.0552699,  0.0252681,
            -0.0562072, -0.123496,  -0.153616, -0.032874,  -0.195349,   0.0192675,   -0.108636,
            0.098927,   -0.140733,  0.162602,  0.0143099,  -0.0455534,  0.0151574,   -0.102509,
            -0.0372696, 0.252296,   -0.144544, 0.00496085, 0.0662588,   -0.048577,   -0.187329,
            0.0855831,  -0.0171894, -0.140202, 0.0828391,  -0.1073,     -0.150145,   0.015065,
            -0.192699,  -0.112764,  -0.120496, 0.155754,   0.148256,    0.208491,    0.348432,
            0.0291103,  0.230275,   -0.165194, -0.0372928, 0.273786,    -0.100877,   -0.0458544,
            -0.0401315, 0.0737483,  -0.064505, 0.136898,   0.00160891,  -0.184812,   0.147774,
            -0.021205,  -0.125423,  0.0206439, -0.187097,  -0.0051453,  -0.0767618,  -0.0735348,
            -0.0826436, 0.214159,   0.262295,  0.0247127,  0.14472};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // 3 args, 1 actv func
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});
        mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(
                      std::vector<migraphx::operation>{migraphx::make_op("sigmoid")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", 0}}),
            seq,
            w,
            r);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{
            0.227861, 0.328562, 0.277867, 0.272945, 0.204389, 0.296123, 0.223834, 0.311113,
            0.424666, 0.173974, 0.40628,  0.286631, 0.246078, 0.199709, 0.303753, 0.301178,
            0.264634, 0.304661, 0.349371, 0.288934, 0.405483, 0.445586, 0.515814, 0.473186,
            0.339438, 0.29655,  0.331832, 0.242338, 0.409384, 0.236272, 0.306045, 0.26269,
            0.261246, 0.334357, 0.23622,  0.245288, 0.301937, 0.264893, 0.254353, 0.269231,
            0.359258, 0.400097, 0.288884, 0.247329, 0.276519, 0.264249, 0.1769,   0.23213,
            0.374123, 0.283167, 0.377129, 0.245726, 0.444712, 0.203168, 0.411446, 0.269965,
            0.172792, 0.296224, 0.17319,  0.352547, 0.310306, 0.262902, 0.276964, 0.295002,
            0.373802, 0.366785, 0.419791, 0.393216, 0.262827, 0.371441, 0.369022, 0.298262,
            0.450186, 0.263538, 0.402895, 0.216177, 0.267257, 0.342535, 0.257797, 0.268563,
            0.193043, 0.275645, 0.167678, 0.350889, 0.334143, 0.309444, 0.174822, 0.251634,
            0.244564, 0.214386, 0.185994, 0.226699, 0.28445,  0.376092, 0.338326, 0.259502};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // 3 args, 2 actv func
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto hs  = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                  {"actv_func",
                   migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                       migraphx::make_op("tanh")})},
                  {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                  {"clip", clip},
                  {"input_forget", 0}}),
            seq,
            w,
            r);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{
            -0.165194, -0.0372928,  0.273786,    -0.100877,  -0.0458544, -0.0401315,
            0.0737483, -0.064505,   0.136898,    0.00160891, -0.184812,  0.147774,
            -0.162851, -0.102647,   -0.113827,   -0.142818,  0.0513685,  0.0547876,
            0.0201981, -0.00808453, -0.00520328, 0.0945081,  0.264123,   0.410805};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // 3 args, 4 actv func
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto hs  = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                  {"actv_func",
                   migraphx::to_value(
                      std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                        migraphx::make_op("tanh"),
                                                        migraphx::make_op("tanh"),
                                                        migraphx::make_op("sigmoid")})},
                  {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                  {"clip", clip},
                  {"input_forget", 0}}),
            seq,
            w,
            r);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{
            -0.165194, -0.0372928, 0.273786, -0.100877,  -0.0458544, -0.0401315,
            0.0737483, -0.064505,  0.136898, 0.00160891, -0.184812,  0.147774,
            0.246078,  0.199709,   0.303753, 0.301178,   0.264634,   0.304661,
            0.349371,  0.288934,   0.405483, 0.445586,   0.515814,   0.473186};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // 3 args, 5 actv func
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto hs  = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                  {"actv_func",
                   migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                       migraphx::make_op("tanh"),
                                                                       migraphx::make_op("tanh"),
                                                                       migraphx::make_op("sigmoid"),
                                                                       migraphx::make_op("tanh")})},
                  {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                  {"clip", clip},
                  {"input_forget", 0}}),
            seq,
            w,
            r);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{
            -0.165194, -0.0372928,  0.273786,    -0.100877,  -0.0458544, -0.0401315,
            0.0737483, -0.064505,   0.136898,    0.00160891, -0.184812,  0.147774,
            -0.162851, -0.102647,   -0.113827,   -0.142818,  0.0513685,  0.0547876,
            0.0201981, -0.00808453, -0.00520328, 0.0945081,  0.264123,   0.410805};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }

    // 3 args, 6 actv func
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_literal(migraphx::literal{in_shape, input_data});
        auto w   = mm->add_literal(migraphx::literal{w_shape, w_data});
        auto r   = mm->add_literal(migraphx::literal{r_shape, r_data});
        auto hs  = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                  {"actv_func",
                   migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                       migraphx::make_op("tanh"),
                                                                       migraphx::make_op("tanh"),
                                                                       migraphx::make_op("sigmoid"),
                                                                       migraphx::make_op("tanh"),
                                                                       migraphx::make_op("tanh")})},
                  {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                  {"clip", clip},
                  {"input_forget", 0}}),
            seq,
            w,
            r);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
        p.compile(migraphx::make_target("ref"));
        auto hs_concat = p.eval({}).back();
        std::vector<float> output_data;
        hs_concat.visit([&](auto output) { output_data.assign(output.begin(), output.end()); });
        std::vector<float> output_data_gold{
            -0.165194, -0.0372928,  0.273786,    -0.100877,  -0.0458544, -0.0401315,
            0.0737483, -0.064505,   0.136898,    0.00160891, -0.184812,  0.147774,
            -0.162851, -0.102647,   -0.113827,   -0.142818,  0.0513685,  0.0547876,
            0.0201981, -0.00808453, -0.00520328, 0.0945081,  0.264123,   0.410805};
        EXPECT(migraphx::verify::verify_rms_range(output_data, output_data_gold));
    }
}

TEST_CASE(lstm_fp16)
{
    std::size_t batch_size  = 3;
    std::size_t seq_len     = 4;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    std::vector<float> w_data{
        0.1236,  -0.3942, 0.4149,  0.0795,  0.4934,  -0.2858, 0.2602,  -0.3098, 0.0567,  0.3344,
        0.3607,  -0.0551, 0.4952,  0.3799,  0.0630,  -0.3532, 0.0023,  -0.0592, 0.4267,  0.2382,
        -0.0784, -0.0032, -0.2476, -0.0206, -0.4963, 0.4837,  0.0827,  0.0123,  -0.1203, -0.0279,
        -0.0049, 0.4721,  -0.3564, -0.1286, 0.4090,  -0.0504, 0.0575,  -0.2138, 0.1071,  0.1976,
        -0.0758, 0.0139,  -0.0761, 0.3991,  -0.2965, -0.4845, -0.1496, 0.3285};

    std::vector<float> r_data{
        0.1237,  0.1229,  -0.0766, -0.1144, -0.1186, 0.2922,  0.2478,  0.3159,  -0.0522, 0.1685,
        -0.4621, 0.1728,  0.0670,  -0.2458, -0.3835, -0.4589, -0.3109, 0.4908,  -0.0133, -0.1858,
        -0.0590, -0.0347, -0.2353, -0.0671, -0.3812, -0.0004, -0.1432, 0.2406,  0.1033,  -0.0265,
        -0.3902, 0.0755,  0.3733,  0.4383,  -0.3140, 0.2537,  -0.1818, -0.4127, 0.3506,  0.2562,
        0.2926,  0.1620,  -0.4849, -0.4861, 0.4426,  0.2106,  -0.0005, 0.4418,  -0.2926, -0.3100,
        0.1500,  -0.0362, -0.3801, -0.0065, -0.0631, 0.1277,  0.2315,  0.4087,  -0.3963, -0.4161,
        -0.2169, -0.1344, 0.3468,  -0.2260};

    std::vector<float> bias_data{0.0088,  0.1183,  0.1642,  -0.2631, -0.1330, -0.4008, 0.3881,
                                 -0.4407, -0.2760, 0.1274,  -0.0083, -0.2885, 0.3949,  -0.0182,
                                 0.4445,  0.3477,  0.2266,  0.3423,  -0.0674, -0.4067, 0.0807,
                                 0.1109,  -0.2036, 0.1782,  -0.2467, -0.0730, -0.4216, 0.0316,
                                 -0.3025, 0.3637,  -0.3181, -0.4655};

    std::vector<float> input_data{
        -0.5516, 0.2391, -1.6951, -0.4313, -0.9730, -0.2005, 2.3930,  -0.5221, -0.1331,
        -0.0910, 1.2122, -0.1952, 0.4661,  0.6494,  2.1332,  -1.0972, 0.9816,  0.1122,
        0.3577,  1.3508, -0.5366, 1.7449,  0.5483,  -0.0701, -0.4100, -2.2344, 0.3685,
        0.4583,  2.3794, 1.0372,  -0.8887, 0.7892,  -0.4012, -0.2818, -2.3374, 1.5310};

    std::vector<float> ih_data{1.9104,
                               -1.9004,
                               0.3337,
                               0.5741,
                               0.5671,
                               0.0458,
                               0.4514,
                               -0.8968,
                               -0.9201,
                               0.1962,
                               0.5771,
                               -0.5332};

    std::vector<float> ic_data{0.9569,
                               -0.5981,
                               1.1312,
                               1.0945,
                               1.1055,
                               -0.1212,
                               -0.9097,
                               0.7831,
                               -1.6991,
                               -1.9498,
                               -1.2567,
                               -0.4114};

    std::vector<float> pph_data{1.84369764,
                                0.68413646,
                                -0.44892886,
                                -1.50904413,
                                0.3860796,
                                -0.52186625,
                                1.08474445,
                                -1.80867321,
                                1.32594529,
                                0.4336262,
                                -0.83699064,
                                0.49162736};

    float clip = 0.0f;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape ic_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size}};

    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto seq  = mm->add_literal(migraphx::literal{in_shape, input_data});
    auto w    = mm->add_literal(migraphx::literal{w_shape, w_data});
    auto r    = mm->add_literal(migraphx::literal{r_shape, r_data});
    auto bias = mm->add_literal(migraphx::literal{b_shape, bias_data});
    auto ih   = mm->add_literal(migraphx::literal{ih_shape, ih_data});
    auto ic   = mm->add_literal(migraphx::literal{ic_shape, ic_data});
    auto und  = mm->add_instruction(migraphx::make_op("undefined"));

    auto seq_half = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), seq);
    auto w_half = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), w);
    auto r_half = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), r);
    auto bias_half = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), bias);
    auto ih_half = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), ih);
    auto ic_half = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), ic);
    auto und_half = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), und);

    mm->add_instruction(
        migraphx::make_op(
            "lstm",
            {{"hidden_size", hidden_size},
             {"actv_func",
              migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                  migraphx::make_op("tanh"),
                                                                  migraphx::make_op("tanh")})},
             {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
             {"clip", clip},
             {"input_forget", 0}}),
        seq_half,
        w_half,
        r_half,
        bias_half,
        und_half,
        ih_half,
        ic_half,
        und_half);
    p.compile(migraphx::make_target("ref"));

    auto hs_concat = p.eval({}).back();
    std::vector<float> hs_data;
    hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

    std::vector<float> hs_data_gold{
        0.0417273, -0.272355,  0.206765,   0.223879,    0.138193,   -0.0322939, -0.0891815,
        0.15773,   0.19139,    -0.127708,  -0.409371,   -0.136186,  0.0742487,  -0.0800085,
        0.259897,  0.0670196,  0.184266,   0.0610048,   -0.138041,  0.0963885,  0.0213755,
        -0.146027, -0.0324509, -0.0620429, -0.00532985, 0.0440265,  0.29654,    -0.0463156,
        0.0498799, 0.125772,   0.0533032,  -0.131413,   0.0988431,  -0.018085,  -0.159434,
        0.030266,  -0.0847427, 0.0874114,  0.304256,    -0.0585745, -0.0223018, 0.131113,
        0.135643,  -0.0566208, 0.142701,   0.0342236,   -0.198664,  0.0702607};
    EXPECT(migraphx::verify::verify_rms_range(hs_data, hs_data_gold, 5e4));
}
