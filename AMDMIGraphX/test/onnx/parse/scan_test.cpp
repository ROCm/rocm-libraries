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

#include <migraphx/common.hpp>
#include <migraphx/literal.hpp>
#include <onnx_test.hpp>

TEST_CASE(scan_test)
{
    namespace mgx = migraphx;
    migraphx::program prog;
    auto* mm        = prog.get_main_module();
    auto init_state = mm->add_parameter("init_state", mgx::shape{mgx::shape::float_type, {2, 2}});
    auto scan_ins1  = mm->add_parameter("scan_ins1", mgx::shape{mgx::shape::float_type, {2, 3, 2}});
    auto scan_ins2  = mm->add_parameter("scan_ins2", mgx::shape{mgx::shape::float_type, {3, 1}});

    auto* body  = prog.create_module("Scan_3_scan");
    auto iter   = body->add_parameter("iter", mgx::shape{mgx::shape::int64_type});
    auto cond   = body->add_parameter("cond", mgx::shape{mgx::shape::bool_type});
    auto sum_in = body->add_parameter("state_var0", mgx::shape{mgx::shape::float_type, {2, 2}});

    auto scan_in2 = body->add_instruction(
        mgx::make_op("scan_slice", {{"axis", 0}, {"direction", 1}}), scan_ins2, iter);
    scan_in2      = body->add_instruction(mgx::make_op("squeeze", {{"axes", {0}}}), scan_in2);
    auto scan_in1 = body->add_instruction(
        mgx::make_op("scan_slice", {{"axis", 1}, {"direction", 0}}), scan_ins1, iter);
    scan_in1 = body->add_instruction(mgx::make_op("squeeze", {{"axes", {1}}}), scan_in1);

    auto add1       = mgx::add_common_op(*body, mgx::make_op("add"), {sum_in, scan_in1});
    auto add2       = mgx::add_common_op(*body, mgx::make_op("add"), {add1, scan_in2});
    auto id         = body->add_instruction(mgx::make_op("identity"), add2);
    auto reduce_sum = body->add_instruction(mgx::make_op("reduce_sum", {{"axes", {0}}}), add2);
    reduce_sum      = body->add_instruction(mgx::make_op("squeeze", {{"axes", {0}}}), reduce_sum);
    body->add_return({cond, add2, id, reduce_sum});

    auto iter_lit = mm->add_literal(mgx::literal{mgx::shape{mgx::shape::int64_type}, {3}});
    auto cond_lit = mm->add_literal(mgx::literal{mgx::shape{mgx::shape::bool_type}, {true}});
    auto loop     = mm->add_instruction(
        mgx::make_op("loop", {{"max_iterations", 3}, {"scan_output_directions", {1, 1}}}),
        {iter_lit, cond_lit, init_state},
        {body});

    auto final_state = mm->add_instruction(mgx::make_op("get_tuple_elem", {{"index", 0}}), loop);
    auto scan_outs1  = mm->add_instruction(mgx::make_op("get_tuple_elem", {{"index", 1}}), loop);
    scan_outs1 =
        mm->add_instruction(mgx::make_op("transpose", {{"permutation", {1, 2, 0}}}), scan_outs1);
    auto scan_outs2 = mm->add_instruction(mgx::make_op("get_tuple_elem", {{"index", 2}}), loop);
    scan_outs2 =
        mm->add_instruction(mgx::make_op("transpose", {{"permutation", {1, 0}}}), scan_outs2);
    mm->add_return({final_state, scan_outs1, scan_outs2});

    auto prog_gold = read_onnx("scan_test6.onnx");
    EXPECT(prog == prog_gold);
}

TEST_CASE(scan_invalid_input_axes_len_test)
{
    EXPECT(test::throws<migraphx::exception>(
        [] { read_onnx("scan_invalid_input_axes_len_test.onnx"); }, "scan_input_axes"));
}

TEST_CASE(scan_invalid_input_dirs_len_test)
{
    EXPECT(test::throws<migraphx::exception>(
        [] { read_onnx("scan_invalid_input_dirs_len_test.onnx"); }, "scan_input_directions"));
}

TEST_CASE(scan_invalid_output_axes_len_test)
{
    EXPECT(test::throws<migraphx::exception>(
        [] { read_onnx("scan_invalid_output_axes_len_test.onnx"); }, "scan_output_axes"));
}

TEST_CASE(scan_invalid_output_dirs_len_test)
{
    EXPECT(test::throws<migraphx::exception>(
        [] { read_onnx("scan_invalid_output_dirs_len_test.onnx"); }, "scan_output_directions"));
}

TEST_CASE(scan_invalid_input_axes_vals_test)
{
    EXPECT(test::throws<migraphx::exception>(
        [] { read_onnx("scan_invalid_input_axes_vals_test.onnx"); }, "scan_input_axes"));
}

TEST_CASE(scan_invalid_input_dirs_vals_test)
{
    EXPECT(test::throws<migraphx::exception>(
        [] { read_onnx("scan_invalid_input_dirs_vals_test.onnx"); }, "scan_input_directions"));
}

TEST_CASE(scan_invalid_output_axes_vals_test)
{
    EXPECT(test::throws<migraphx::exception>(
        [] { read_onnx("scan_invalid_output_axes_vals_test.onnx"); }, "scan_output_axes"));
}

TEST_CASE(scan_invalid_output_dirs_vals_test)
{
    EXPECT(test::throws<migraphx::exception>(
        [] { read_onnx("scan_invalid_output_dirs_vals_test.onnx"); }, "scan_output_directions"));
}

TEST_CASE(scan_arg_count_mismatch_test)
{
    EXPECT(test::throws([] { read_onnx("scan_arg_count_mismatch_test.onnx"); }));
}

TEST_CASE(scan_arg_shapes_mismatch_test)
{
    EXPECT(test::throws([] { read_onnx("scan_arg_shapes_mismatch_test.onnx"); }));
}

TEST_CASE(scan_input_axes_lens_mismatch_test)
{
    EXPECT(test::throws([] { read_onnx("scan_input_axes_lens_mismatch_test.onnx"); }));
}
