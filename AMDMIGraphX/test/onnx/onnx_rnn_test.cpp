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
#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/serialize.hpp>

#include <migraphx/make_op.hpp>
#include <onnx_test.hpp>
#include "test.hpp"

static migraphx::program read_rnn_onnx(const std::string& name, bool eliminate_deadcode = true)
{
    auto prog = read_onnx(name);
    auto* mm  = prog.get_main_module();
    if(eliminate_deadcode)
        migraphx::run_passes(*mm, {migraphx::dead_code_elimination{}});

    // remove the last identity instruction
    auto last_ins = std::prev(mm->end());
    if(last_ins->name() == "@return")
    {
        mm->remove_instruction(last_ins);
    }

    return prog;
}

TEST_CASE(rnn_test_bidirectional)
{
    std::size_t sl = 5;  // sequence len
    std::size_t bs = 3;  // batch size
    std::size_t hs = 20; // hidden size
    std::size_t is = 10; // input size
    std::size_t nd = 2;  // num directions
    float clip     = 0.0f;
    migraphx::shape seq_shape{migraphx::shape::float_type, {sl, bs, is}};
    migraphx::shape w_shape{migraphx::shape::float_type, {nd, hs, is}};
    migraphx::shape r_shape{migraphx::shape::float_type, {nd, hs, hs}};
    migraphx::shape bias_shape{migraphx::shape::float_type, {nd, 2 * hs}};
    migraphx::shape sl_shape{migraphx::shape::int32_type, {bs}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {nd, bs, hs}};

    migraphx::program p;
    auto* mm = p.get_main_module();

    auto seq     = mm->add_parameter("seq", seq_shape);
    auto w       = mm->add_parameter("w", w_shape);
    auto r       = mm->add_parameter("r", r_shape);
    auto bias    = mm->add_parameter("bias", bias_shape);
    auto seq_len = mm->add_parameter("seq_len", sl_shape);
    auto ih      = mm->add_parameter("h0", ih_shape);

    auto out_hs = mm->add_instruction(
        migraphx::make_op(
            "rnn",
            {{"hidden_size", hs},
             {"actv_func",
              migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh"),
                                                                  migraphx::make_op("sigmoid")})},
             {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
             {"clip", clip}}),
        seq,
        w,
        r,
        bias,
        seq_len,
        ih);
    mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
    auto prog = read_rnn_onnx("onnx_rnn_bi.onnx");

    EXPECT(p == prog);
}

TEST_CASE(rnn_test_bidirectional_layout)
{
    std::size_t sl = 5;  // sequence len
    std::size_t bs = 3;  // batch size
    std::size_t hs = 20; // hidden size
    std::size_t is = 10; // input size
    std::size_t nd = 2;  // num directions
    float clip     = 0.0f;
    migraphx::shape seq_shape{migraphx::shape::float_type, {bs, sl, is}};
    migraphx::shape w_shape{migraphx::shape::float_type, {nd, hs, is}};
    migraphx::shape r_shape{migraphx::shape::float_type, {nd, hs, hs}};
    migraphx::shape bias_shape{migraphx::shape::float_type, {nd, 2 * hs}};
    migraphx::shape sl_shape{migraphx::shape::int32_type, {bs}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {bs, nd, hs}};

    migraphx::program p;
    auto* mm = p.get_main_module();

    auto seq     = mm->add_parameter("seq", seq_shape);
    auto w       = mm->add_parameter("w", w_shape);
    auto r       = mm->add_parameter("r", r_shape);
    auto bias    = mm->add_parameter("bias", bias_shape);
    auto seq_len = mm->add_parameter("seq_len", sl_shape);
    auto ih      = mm->add_parameter("h0", ih_shape);

    std::vector<int64_t> perm{1, 0, 2};
    seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
    ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

    auto out_hs = mm->add_instruction(
        migraphx::make_op(
            "rnn",
            {{"hidden_size", hs},
             {"actv_func",
              migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh"),
                                                                  migraphx::make_op("sigmoid")})},
             {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
             {"clip", clip}}),
        seq,
        w,
        r,
        bias,
        seq_len,
        ih);
    auto last_output = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
    std::vector<int64_t> perm_hid{2, 0, 1, 3};
    out_hs =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}), out_hs);
    mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_output);
    auto prog = read_rnn_onnx("rnn_bi_layout_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(rnn_test_one_direction)
{
    std::size_t sl = 5;  // sequence len
    std::size_t bs = 3;  // batch size
    std::size_t hs = 20; // hidden size
    std::size_t is = 10; // input size
    std::size_t nd = 1;  // num directions
    float clip     = 0.0f;
    migraphx::shape seq_shape{migraphx::shape::float_type, {sl, bs, is}};
    migraphx::shape w_shape{migraphx::shape::float_type, {nd, hs, is}};
    migraphx::shape r_shape{migraphx::shape::float_type, {nd, hs, hs}};
    migraphx::shape bias_shape{migraphx::shape::float_type, {nd, 2 * hs}};
    migraphx::shape sl_shape{migraphx::shape::int32_type, {bs}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {nd, bs, hs}};

    // forward
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto ih      = mm->add_parameter("h0", ih_shape);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_rnn_forward.onnx");

        EXPECT(p == prog);
    }

    // forward, default activation functions
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_parameter("seq", seq_shape);
        auto w   = mm->add_parameter("w", w_shape);
        auto r   = mm->add_parameter("r", r_shape);
        auto und = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            und,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("rnn_f_default_af_test.onnx");

        EXPECT(p == prog);
    }

    // reverse
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto ih      = mm->add_parameter("h0", ih_shape);
        auto out_hs  = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hs},
                  {"actv_func",
                   migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                  {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                  {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_rnn_reverse.onnx");

        EXPECT(p == prog);
    }

    // 3 arguments
    {
        migraphx::program p;
        auto* mm    = p.get_main_module();
        auto seq    = mm->add_parameter("seq", seq_shape);
        auto w      = mm->add_parameter("w", w_shape);
        auto r      = mm->add_parameter("r", r_shape);
        auto und    = mm->add_instruction(migraphx::make_op("undefined"));
        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            und,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_rnn_3args.onnx");

        EXPECT(p == prog);
    }

    // 5 arguments
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto und     = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_rnn_5args.onnx");

        EXPECT(p == prog);
    }
}

TEST_CASE(rnn_test_one_direction_layout)
{
    std::size_t sl = 5;  // sequence len
    std::size_t bs = 3;  // batch size
    std::size_t hs = 20; // hidden size
    std::size_t is = 10; // input size
    std::size_t nd = 1;  // num directions
    float clip     = 0.0f;
    migraphx::shape seq_shape{migraphx::shape::float_type, {bs, sl, is}};
    migraphx::shape w_shape{migraphx::shape::float_type, {nd, hs, is}};
    migraphx::shape r_shape{migraphx::shape::float_type, {nd, hs, hs}};
    migraphx::shape bias_shape{migraphx::shape::float_type, {nd, 2 * hs}};
    migraphx::shape sl_shape{migraphx::shape::int32_type, {bs}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {bs, nd, hs}};

    // forward
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto ih      = mm->add_parameter("h0", ih_shape);

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih);
        auto last_output = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_output);
        auto prog = read_rnn_onnx("rnn_f_layout_test.onnx");

        EXPECT(p == prog);
    }

    // reverse
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto ih      = mm->add_parameter("h0", ih_shape);

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih);
        auto last_output = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_output);
        auto prog = read_rnn_onnx("rnn_r_layout_test.onnx");

        EXPECT(p == prog);
    }

    // 3 arguments
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_parameter("seq", seq_shape);
        auto w   = mm->add_parameter("w", w_shape);
        auto r   = mm->add_parameter("r", r_shape);
        auto und = mm->add_instruction(migraphx::make_op("undefined"));

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            und,
            und,
            und);
        auto last_output = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_output);
        auto prog = read_rnn_onnx("rnn_r_3arg_layout_test.onnx");

        EXPECT(p == prog);
    }

    // 5 arguments
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto und     = mm->add_instruction(migraphx::make_op("undefined"));

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            und);
        auto last_output = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_output);
        auto prog = read_rnn_onnx("rnn_f_5arg_layout_test.onnx");

        EXPECT(p == prog);
    }
}

TEST_CASE(rnn_invalid_af_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("rnn_bi_1af_test.onnx"); }));
}

TEST_CASE(gru_test)
{
    std::size_t sl = 5;  // sequence len
    std::size_t bs = 3;  // batch size
    std::size_t hs = 20; // hidden size
    std::size_t is = 10; // input size
    std::size_t nd = 2;  // num directions
    float clip     = 0.0f;
    // forward
    {
        nd = 1;
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto seq =
            mm->add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            mm->add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            mm->add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            mm->add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih =
            mm->add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {nd, bs, hs}});

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{
                      migraphx::make_op("tanh"), migraphx::make_op("sigmoid")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_gru_forward.onnx");

        EXPECT(p == prog);
    }

    // reverse
    {
        nd = 1;
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto seq =
            mm->add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            mm->add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            mm->add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            mm->add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih =
            mm->add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {nd, bs, hs}});

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{
                      migraphx::make_op("tanh"), migraphx::make_op("sigmoid")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_gru_reverse.onnx");

        EXPECT(p == prog);
    }

    // bidirectional
    {
        nd = 2;
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto seq =
            mm->add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            mm->add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            mm->add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            mm->add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih =
            mm->add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {nd, bs, hs}});

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("relu"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_gru_bi.onnx");

        EXPECT(p == prog);
    }
}

TEST_CASE(gru_layout_test)
{
    std::size_t sl = 5;  // sequence len
    std::size_t bs = 3;  // batch size
    std::size_t hs = 20; // hidden size
    std::size_t is = 10; // input size
    std::size_t nd = 2;  // num directions
    float clip     = 0.0f;
    // forward
    {
        nd = 1;
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto seq =
            mm->add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {bs, sl, is}});
        auto w =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            mm->add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            mm->add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            mm->add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih =
            mm->add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {bs, nd, hs}});

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{
                      migraphx::make_op("tanh"), migraphx::make_op("sigmoid")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih);
        auto last_output = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_output);
        auto prog = read_rnn_onnx("gru_f_layout_test.onnx");

        EXPECT(p == prog);
    }

    // reverse
    {
        nd = 1;
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto seq =
            mm->add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {bs, sl, is}});
        auto w =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            mm->add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            mm->add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            mm->add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih =
            mm->add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {bs, nd, hs}});

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{
                      migraphx::make_op("tanh"), migraphx::make_op("sigmoid")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih);
        auto last_output = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_output);
        auto prog = read_rnn_onnx("gru_r_layout_test.onnx");

        EXPECT(p == prog);
    }

    // bidirectional
    {
        nd = 2;
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto seq =
            mm->add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {bs, sl, is}});
        auto w =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            mm->add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            mm->add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            mm->add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih =
            mm->add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {bs, nd, hs}});

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("relu"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih);
        auto last_output = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_output);
        auto prog = read_rnn_onnx("gru_bi_layout_test.onnx");

        EXPECT(p == prog);
    }
}

TEST_CASE(gru_test_args)
{
    std::size_t sl = 5;  // sequence len
    std::size_t bs = 3;  // batch size
    std::size_t hs = 20; // hidden size
    std::size_t is = 10; // input size
    std::size_t nd = 2;  // num directions
    float clip     = 0.0f;

    // 3 arguments
    {
        nd = 1;
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto seq =
            mm->add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            mm->add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto und    = mm->add_instruction(migraphx::make_op("undefined"));
        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{
                      migraphx::make_op("tanh"), migraphx::make_op("sigmoid")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            und,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_gru_3arg.onnx");

        EXPECT(p == prog);
    }

    // 4 arguments
    {
        nd = 1;
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto seq =
            mm->add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            mm->add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            mm->add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto und = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("relu"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_gru_4arg.onnx");

        EXPECT(p == prog);
    }

    // 5 arguments
    {
        nd = 2;
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto seq =
            mm->add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            mm->add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            mm->add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            mm->add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto und = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("relu"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_gru_5arg.onnx");

        EXPECT(p == prog);
    }
}

TEST_CASE(gru_test_args_layout)
{
    std::size_t sl = 5;  // sequence len
    std::size_t bs = 3;  // batch size
    std::size_t hs = 20; // hidden size
    std::size_t is = 10; // input size
    std::size_t nd = 2;  // num directions
    float clip     = 0.0f;

    // 3 arguments
    {
        nd = 1;
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto seq =
            mm->add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {bs, sl, is}});
        auto w =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            mm->add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto und = mm->add_instruction(migraphx::make_op("undefined"));

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{
                      migraphx::make_op("tanh"), migraphx::make_op("sigmoid")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            und,
            und,
            und);
        auto last_output = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_output);
        auto prog = read_rnn_onnx("gru_f_3arg_layout_test.onnx");

        EXPECT(p == prog);
    }

    // 4 arguments
    {
        nd = 1;
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto seq =
            mm->add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {bs, sl, is}});
        auto w =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            mm->add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            mm->add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto und = mm->add_instruction(migraphx::make_op("undefined"));

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("relu"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            und,
            und);
        auto last_output = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_output);
        auto prog = read_rnn_onnx("gru_r_4arg_layout_test.onnx");

        EXPECT(p == prog);
    }

    // 5 arguments
    {
        nd = 2;
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto seq =
            mm->add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {bs, sl, is}});
        auto w =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            mm->add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            mm->add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            mm->add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto und = mm->add_instruction(migraphx::make_op("undefined"));

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("relu"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"linear_before_reset", 1}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            und);
        auto last_output = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_output);
        auto prog = read_rnn_onnx("gru_bi_5arg_layout_test.onnx");

        EXPECT(p == prog);
    }
}

TEST_CASE(gru_test_actv_funcs)
{
    std::size_t sl = 5;  // sequence len
    std::size_t bs = 3;  // batch size
    std::size_t hs = 20; // hidden size
    std::size_t is = 10; // input size
    std::size_t nd = 2;  // num directions
    float clip     = 0.0f;
    // bidirection, 0 actv function
    {
        nd = 2;
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto seq =
            mm->add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            mm->add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            mm->add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            mm->add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih =
            mm->add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {nd, bs, hs}});

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_gru_bi_0.onnx");

        EXPECT(p == prog);
    }

    // bidirection, 1 actv function
    {
        nd = 2;
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto seq =
            mm->add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            mm->add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            mm->add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            mm->add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih =
            mm->add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {nd, bs, hs}});

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(
                      std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                       migraphx::make_op("sigmoid"),
                                                       migraphx::make_op("sigmoid"),
                                                       migraphx::make_op("sigmoid")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_gru_bi_1.onnx");

        EXPECT(p == prog);
    }

    // bidirection, 2 actv functions
    {
        nd = 2;
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto seq =
            mm->add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            mm->add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            mm->add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            mm->add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih =
            mm->add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {nd, bs, hs}});

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_gru_bi_2.onnx");

        EXPECT(p == prog);
    }

    // forward, 0 actv function
    {
        nd = 1;
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto seq =
            mm->add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            mm->add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            mm->add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            mm->add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih =
            mm->add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {nd, bs, hs}});

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_gru_forward_0.onnx");

        EXPECT(p == prog);
    }

    // reverse, 1 actv function
    {
        nd = 1;
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto seq =
            mm->add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            mm->add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            mm->add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            mm->add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih =
            mm->add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {nd, bs, hs}});

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("relu"),
                                                                      migraphx::make_op("relu")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_gru_reverse_1.onnx");

        EXPECT(p == prog);
    }
}

TEST_CASE(gru_invalid_af_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("gru_f_1af_test.onnx"); }));
}

TEST_CASE(lstm_forward)
{
    std::size_t sl   = 5;  // sequence len
    std::size_t bs   = 3;  // batch size
    std::size_t hs   = 20; // hidden size
    std::size_t is   = 10; // input size
    std::size_t nd   = 1;  // num directions
    float clip       = 0.0f;
    int input_forget = 1;
    migraphx::shape seq_shape{migraphx::shape::float_type, {sl, bs, is}};
    migraphx::shape w_shape{migraphx::shape::float_type, {nd, 4 * hs, is}};
    migraphx::shape r_shape{migraphx::shape::float_type, {nd, 4 * hs, hs}};
    migraphx::shape bias_shape{migraphx::shape::float_type, {nd, 8 * hs}};
    migraphx::shape sl_shape{migraphx::shape::int32_type, {bs}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {nd, bs, hs}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {nd, 3 * hs}};
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto ih      = mm->add_parameter("h0", ih_shape);
        auto ic      = mm->add_parameter("c0", ih_shape);
        auto pph     = mm->add_parameter("pph", pph_shape);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih,
            ic,
            pph);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_forward.onnx");

        EXPECT(p == prog);
    }

    // 3 args
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_parameter("seq", seq_shape);
        auto w   = mm->add_parameter("w", w_shape);
        auto r   = mm->add_parameter("r", r_shape);
        auto und = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            und,
            und,
            und,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_f3args.onnx");

        EXPECT(p == prog);
    }

    // 3 args, hs output
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_parameter("seq", seq_shape);
        auto w   = mm->add_parameter("w", w_shape);
        auto r   = mm->add_parameter("r", r_shape);
        auto und = mm->add_instruction(migraphx::make_op("undefined"));

        mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            und,
            und,
            und,
            und,
            und);
        auto prog = read_rnn_onnx("onnx_lstm_hs.onnx");

        EXPECT(p == prog);
    }

    // 3 args, last output
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_parameter("seq", seq_shape);
        auto w   = mm->add_parameter("w", w_shape);
        auto r   = mm->add_parameter("r", r_shape);
        auto und = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            und,
            und,
            und,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_last.onnx");

        EXPECT(p == prog);
    }

    // 3 args, cell output
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_parameter("seq", seq_shape);
        auto w   = mm->add_parameter("w", w_shape);
        auto r   = mm->add_parameter("r", r_shape);
        auto und = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            und,
            und,
            und,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_cell_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_cell.onnx");

        EXPECT(p == prog);
    }

    // 4 args
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_parameter("seq", seq_shape);
        auto w    = mm->add_parameter("w", w_shape);
        auto r    = mm->add_parameter("r", r_shape);
        auto bias = mm->add_parameter("bias", bias_shape);
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            und,
            und,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_f4args.onnx");

        EXPECT(p == prog);
    }

    // 5 args
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto und     = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            und,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        mm->add_instruction(migraphx::make_op("rnn_last_cell_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_f5args.onnx");

        EXPECT(p == prog);
    }

    // 6 args
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto ih      = mm->add_parameter("h0", ih_shape);
        auto und     = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        mm->add_instruction(migraphx::make_op("rnn_last_cell_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_f6args.onnx");

        EXPECT(p == prog);
    }

    // 7 args
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto ih      = mm->add_parameter("h0", ih_shape);
        auto ic      = mm->add_parameter("c0", ih_shape);
        auto und     = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih,
            ic,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        mm->add_instruction(migraphx::make_op("rnn_last_cell_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_f7args.onnx");

        EXPECT(p == prog);
    }
}

TEST_CASE(lstm_forward_layout)
{
    std::size_t sl   = 5;  // sequence len
    std::size_t bs   = 3;  // batch size
    std::size_t hs   = 20; // hidden size
    std::size_t is   = 10; // input size
    std::size_t nd   = 1;  // num directions
    float clip       = 0.0f;
    int input_forget = 1;
    migraphx::shape seq_shape{migraphx::shape::float_type, {bs, sl, is}};
    migraphx::shape w_shape{migraphx::shape::float_type, {nd, 4 * hs, is}};
    migraphx::shape r_shape{migraphx::shape::float_type, {nd, 4 * hs, hs}};
    migraphx::shape bias_shape{migraphx::shape::float_type, {nd, 8 * hs}};
    migraphx::shape sl_shape{migraphx::shape::int32_type, {bs}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {bs, nd, hs}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {nd, 3 * hs}};

    // 8 args, hs and last output
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto ih      = mm->add_parameter("h0", ih_shape);
        auto ic      = mm->add_parameter("c0", ih_shape);
        auto pph     = mm->add_parameter("pph", pph_shape);

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);
        ic  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ic);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih,
            ic,
            pph);
        auto last_output = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_output);

        auto prog = read_rnn_onnx("lstm_f_layout_hs_test.onnx");

        EXPECT(p == prog);
    }
    // 8 args, cell output
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto ih      = mm->add_parameter("h0", ih_shape);
        auto ic      = mm->add_parameter("c0", ih_shape);
        auto pph     = mm->add_parameter("pph", pph_shape);

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);
        ic  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ic);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih,
            ic,
            pph);
        auto last_cell = mm->add_instruction(migraphx::make_op("rnn_last_cell_output"), out_hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_cell);
        auto prog = read_rnn_onnx("lstm_f_layout_cell_test.onnx");

        EXPECT(p == prog);
    }
}

// activation functions
TEST_CASE(lstm_forward_actv_func)
{
    std::size_t sl   = 5;  // sequence len
    std::size_t bs   = 3;  // batch size
    std::size_t hs   = 20; // hidden size
    std::size_t is   = 10; // input size
    std::size_t nd   = 1;  // num directions
    float clip       = 0.0f;
    int input_forget = 1;
    migraphx::shape seq_shape{migraphx::shape::float_type, {sl, bs, is}};
    migraphx::shape w_shape{migraphx::shape::float_type, {nd, 4 * hs, is}};
    migraphx::shape r_shape{migraphx::shape::float_type, {nd, 4 * hs, hs}};
    migraphx::shape bias_shape{migraphx::shape::float_type, {nd, 8 * hs}};
    migraphx::shape sl_shape{migraphx::shape::int32_type, {bs}};
    // no activation function specified
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_parameter("seq", seq_shape);
        auto w   = mm->add_parameter("w", w_shape);
        auto r   = mm->add_parameter("r", r_shape);
        // auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto und = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            und,
            und,
            und,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_f0af.onnx");

        EXPECT(p == prog);
    }

    // 1 activation function specified
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_parameter("seq", seq_shape);
        auto w    = mm->add_parameter("w", w_shape);
        auto r    = mm->add_parameter("r", r_shape);
        auto bias = mm->add_parameter("bias", bias_shape);
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(
                      std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                       migraphx::make_op("sigmoid"),
                                                       migraphx::make_op("sigmoid")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            und,
            und,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_f1af.onnx");

        EXPECT(p == prog);
    }

    // 2 non-default activation function specified
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto und     = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(
                      std::vector<migraphx::operation>{migraphx::make_op("tanh"),
                                                       migraphx::make_op("sigmoid"),
                                                       migraphx::make_op("sigmoid")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            und,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        mm->add_instruction(migraphx::make_op("rnn_last_cell_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_f2af.onnx");

        EXPECT(p == prog);
    }
}

TEST_CASE(lstm_reverse)
{
    std::size_t sl   = 5;  // sequence len
    std::size_t bs   = 3;  // batch size
    std::size_t hs   = 20; // hidden size
    std::size_t is   = 10; // input size
    std::size_t nd   = 1;  // num directions
    float clip       = 0.0f;
    int input_forget = 1;
    migraphx::shape seq_shape{migraphx::shape::float_type, {sl, bs, is}};
    migraphx::shape w_shape{migraphx::shape::float_type, {nd, 4 * hs, is}};
    migraphx::shape r_shape{migraphx::shape::float_type, {nd, 4 * hs, hs}};
    migraphx::shape bias_shape{migraphx::shape::float_type, {nd, 8 * hs}};
    migraphx::shape sl_shape{migraphx::shape::int32_type, {bs}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {nd, bs, hs}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {nd, 3 * hs}};
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto ih      = mm->add_parameter("h0", ih_shape);
        auto ic      = mm->add_parameter("c0", ih_shape);
        auto pph     = mm->add_parameter("pph", pph_shape);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih,
            ic,
            pph);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_reverse.onnx");

        EXPECT(p == prog);
    }

    // 5 args
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto und     = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            und,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        mm->add_instruction(migraphx::make_op("rnn_last_cell_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_r5args.onnx");

        EXPECT(p == prog);
    }

    // no activation function specified
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_parameter("seq", seq_shape);
        auto w   = mm->add_parameter("w", w_shape);
        auto r   = mm->add_parameter("r", r_shape);
        auto und = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            und,
            und,
            und,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_r0af.onnx");

        EXPECT(p == prog);
    }
}

TEST_CASE(lstm_reverse_layout)
{
    std::size_t sl   = 5;  // sequence len
    std::size_t bs   = 3;  // batch size
    std::size_t hs   = 20; // hidden size
    std::size_t is   = 10; // input size
    std::size_t nd   = 1;  // num directions
    float clip       = 0.0f;
    int input_forget = 1;
    migraphx::shape seq_shape{migraphx::shape::float_type, {bs, sl, is}};
    migraphx::shape w_shape{migraphx::shape::float_type, {nd, 4 * hs, is}};
    migraphx::shape r_shape{migraphx::shape::float_type, {nd, 4 * hs, hs}};
    migraphx::shape bias_shape{migraphx::shape::float_type, {nd, 8 * hs}};
    migraphx::shape sl_shape{migraphx::shape::int32_type, {bs}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {bs, nd, hs}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {nd, 3 * hs}};

    // 8 args, hs output
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto ih      = mm->add_parameter("h0", ih_shape);
        auto ic      = mm->add_parameter("c0", ih_shape);
        auto pph     = mm->add_parameter("pph", pph_shape);

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);
        ic  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ic);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih,
            ic,
            pph);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs    = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        auto prog = read_rnn_onnx("lstm_r_layout_test.onnx");

        EXPECT(p == prog);
    }

    // 8 args, last and cell output
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto ih      = mm->add_parameter("h0", ih_shape);
        auto ic      = mm->add_parameter("c0", ih_shape);
        auto pph     = mm->add_parameter("pph", pph_shape);

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);
        ic  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ic);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih,
            ic,
            pph);
        auto last_output = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto last_cell   = mm->add_instruction(migraphx::make_op("rnn_last_cell_output"), out_hs);
        last_output = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}),
                                          last_output);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_cell);

        auto prog = read_rnn_onnx("lstm_r_layout_hs_cell_test.onnx");

        EXPECT(p == prog);
    }
}

TEST_CASE(lstm_bidirectional)
{
    std::size_t sl   = 5;  // sequence len
    std::size_t bs   = 3;  // batch size
    std::size_t hs   = 20; // hidden size
    std::size_t is   = 10; // input size
    std::size_t nd   = 2;  // num directions
    float clip       = 0.0f;
    int input_forget = 1;
    migraphx::shape seq_shape{migraphx::shape::float_type, {sl, bs, is}};
    migraphx::shape w_shape{migraphx::shape::float_type, {nd, 4 * hs, is}};
    migraphx::shape r_shape{migraphx::shape::float_type, {nd, 4 * hs, hs}};
    migraphx::shape bias_shape{migraphx::shape::float_type, {nd, 8 * hs}};
    migraphx::shape sl_shape{migraphx::shape::int32_type, {bs}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {nd, bs, hs}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {nd, 3 * hs}};
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto ih      = mm->add_parameter("h0", ih_shape);
        auto ic      = mm->add_parameter("c0", ih_shape);
        auto pph     = mm->add_parameter("pph", pph_shape);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih,
            ic,
            pph);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_bi.onnx");

        EXPECT(p == prog);
    }

    // 3 args
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_parameter("seq", seq_shape);
        auto w   = mm->add_parameter("w", w_shape);
        auto r   = mm->add_parameter("r", r_shape);
        auto und = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            und,
            und,
            und,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_bi3args.onnx");

        EXPECT(p == prog);
    }

    // 4 args
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_parameter("seq", seq_shape);
        auto w    = mm->add_parameter("w", w_shape);
        auto r    = mm->add_parameter("r", r_shape);
        auto bias = mm->add_parameter("bias", bias_shape);
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            und,
            und,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_bi4args.onnx");

        EXPECT(p == prog);
    }

    // 5 args
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto und     = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            und,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_bi5args.onnx");

        EXPECT(p == prog);
    }

    // 6 args
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto ih      = mm->add_parameter("h0", ih_shape);
        auto und     = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_bi6args.onnx");

        EXPECT(p == prog);
    }

    // 7 args
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto ih      = mm->add_parameter("h0", ih_shape);
        auto ic      = mm->add_parameter("c0", ih_shape);
        auto und     = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih,
            ic,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_bi7args.onnx");

        EXPECT(p == prog);
    }
}

TEST_CASE(lstm_bidirectional_layout)
{
    std::size_t sl   = 5;  // sequence len
    std::size_t bs   = 3;  // batch size
    std::size_t hs   = 20; // hidden size
    std::size_t is   = 10; // input size
    std::size_t nd   = 2;  // num directions
    float clip       = 0.0f;
    int input_forget = 1;
    migraphx::shape seq_shape{migraphx::shape::float_type, {bs, sl, is}};
    migraphx::shape w_shape{migraphx::shape::float_type, {nd, 4 * hs, is}};
    migraphx::shape r_shape{migraphx::shape::float_type, {nd, 4 * hs, hs}};
    migraphx::shape bias_shape{migraphx::shape::float_type, {nd, 8 * hs}};
    migraphx::shape sl_shape{migraphx::shape::int32_type, {bs}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {bs, nd, hs}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {nd, 3 * hs}};
    // 0 activation function
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto ih      = mm->add_parameter("h0", ih_shape);
        auto ic      = mm->add_parameter("c0", ih_shape);
        auto pph     = mm->add_parameter("pph", pph_shape);

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);
        ic  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ic);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih,
            ic,
            pph);
        auto last_output = mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        std::vector<int64_t> perm_hid{2, 0, 1, 3};
        out_hs = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm_hid}}),
                                     out_hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_output);
        auto prog = read_rnn_onnx("lstm_bi_layout_last_test.onnx");

        EXPECT(p == prog);
    }
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto ih      = mm->add_parameter("h0", ih_shape);
        auto ic      = mm->add_parameter("c0", ih_shape);
        auto pph     = mm->add_parameter("pph", pph_shape);

        std::vector<int64_t> perm{1, 0, 2};
        seq = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), seq);
        ih  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ih);
        ic  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), ic);

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih,
            ic,
            pph);
        auto last_cell = mm->add_instruction(migraphx::make_op("rnn_last_cell_output"), out_hs);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), last_cell);
        auto prog = read_rnn_onnx("lstm_bi_layout_cell_test.onnx");

        EXPECT(p == prog);
    }
}

TEST_CASE(lstm_bi_actv_funcs)
{
    std::size_t sl   = 5;  // sequence len
    std::size_t bs   = 3;  // batch size
    std::size_t hs   = 20; // hidden size
    std::size_t is   = 10; // input size
    std::size_t nd   = 2;  // num directions
    float clip       = 0.0f;
    int input_forget = 1;
    migraphx::shape seq_shape{migraphx::shape::float_type, {sl, bs, is}};
    migraphx::shape w_shape{migraphx::shape::float_type, {nd, 4 * hs, is}};
    migraphx::shape r_shape{migraphx::shape::float_type, {nd, 4 * hs, hs}};
    migraphx::shape bias_shape{migraphx::shape::float_type, {nd, 8 * hs}};
    migraphx::shape sl_shape{migraphx::shape::int32_type, {bs}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {nd, bs, hs}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {nd, 3 * hs}};

    // 0 activation function
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_parameter("seq", seq_shape);
        auto w   = mm->add_parameter("w", w_shape);
        auto r   = mm->add_parameter("r", r_shape);
        auto und = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            und,
            und,
            und,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_bi0af.onnx");

        EXPECT(p == prog);
    }

    // all activation functions - sigmoid
    {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto seq  = mm->add_parameter("seq", seq_shape);
        auto w    = mm->add_parameter("w", w_shape);
        auto r    = mm->add_parameter("r", r_shape);
        auto bias = mm->add_parameter("bias", bias_shape);
        auto und  = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(
                      std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                       migraphx::make_op("sigmoid"),
                                                       migraphx::make_op("sigmoid"),
                                                       migraphx::make_op("sigmoid"),
                                                       migraphx::make_op("sigmoid"),
                                                       migraphx::make_op("sigmoid")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            und,
            und,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_bi1af.onnx");

        EXPECT(p == prog);
    }

    // all forward direction functions are tanh, default reverse direction
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto und     = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            und,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_bi2af.onnx");

        EXPECT(p == prog);
    }

    // default forward direction, all reverse direction functions are tanh
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto ih      = mm->add_parameter("h0", ih_shape);
        auto und     = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_bi3af.onnx");

        EXPECT(p == prog);
    }

    // no-default forward direction functions, default reverse direction
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto seq     = mm->add_parameter("seq", seq_shape);
        auto w       = mm->add_parameter("w", w_shape);
        auto r       = mm->add_parameter("r", r_shape);
        auto bias    = mm->add_parameter("bias", bias_shape);
        auto seq_len = mm->add_parameter("seq_len", sl_shape);
        auto ih      = mm->add_parameter("h0", ih_shape);
        auto ic      = mm->add_parameter("c0", ih_shape);
        auto und     = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            bias,
            seq_len,
            ih,
            ic,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_bi4af.onnx");

        EXPECT(p == prog);
    }

    // default forward direction, no-default reverse direction functions
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto seq = mm->add_parameter("seq", seq_shape);
        auto w   = mm->add_parameter("w", w_shape);
        auto r   = mm->add_parameter("r", r_shape);
        auto und = mm->add_instruction(migraphx::make_op("undefined"));

        auto out_hs = mm->add_instruction(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hs},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("tanh"),
                                                                      migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip},
                 {"input_forget", input_forget}}),
            seq,
            w,
            r,
            und,
            und,
            und,
            und,
            und);
        mm->add_instruction(migraphx::make_op("rnn_last_hs_output"), out_hs);
        auto prog = read_rnn_onnx("onnx_lstm_bi5af.onnx");

        EXPECT(p == prog);
    }
}

TEST_CASE(lstm_invalid_af_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("lstm_f_1af_test.onnx"); }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
