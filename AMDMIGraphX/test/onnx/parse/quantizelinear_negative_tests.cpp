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

#include <onnx_test.hpp>

TEST_CASE(quantizelinear_too_few_inputs_test)
{
    EXPECT(test::throws([&] { read_onnx("quantizelinear_too_few_inputs_test.onnx"); }));
}

TEST_CASE(quantizelinear_too_many_inputs_test)
{
    EXPECT(test::throws([&] { read_onnx("quantizelinear_too_many_inputs_test.onnx"); }));
}

TEST_CASE(quantizelinear_scales_and_zp_shape_mismatch_test)
{
    EXPECT(
        test::throws([&] { read_onnx("quantizelinear_scales_and_zp_shape_mismatch_test.onnx"); }));
}

TEST_CASE(quantizelinear_output_dtype_and_zp_type_mismatch_test)
{
    EXPECT(test::throws(
        [&] { read_onnx("quantizelinear_output_dtype_and_zp_type_mismatch_test.onnx"); }));
}

TEST_CASE(quantizelinear_per_axis_shape_mismatch_test)
{
    EXPECT(test::throws([&] { read_onnx("quantizelinear_per_axis_shape_mismatch_test.onnx"); }));
}

TEST_CASE(quantizelinear_blocked_x_and_scales_rank_mismatch_test)
{
    EXPECT(test::throws(
        [&] { read_onnx("quantizelinear_blocked_x_and_scales_rank_mismatch_test.onnx"); }));
}

TEST_CASE(quantizelinear_blocked_non_bc_axis_size_mismatch_test)
{
    EXPECT(test::throws(
        [&] { read_onnx("quantizelinear_blocked_non_bc_axis_size_mismatch_test.onnx"); }));
}

TEST_CASE(quantizelinear_blocked_invalid_block_size_test)
{
    EXPECT(test::throws([&] { read_onnx("quantizelinear_blocked_invalid_block_size_test.onnx"); }));
}
