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

TEST_CASE(einsum_missing_equation_negative_test)
{
    EXPECT(test::throws([&] { read_onnx("einsum_missing_equation_negative_test.onnx"); }));
}

TEST_CASE(einsum_multiple_arrows_negative_test)
{
    EXPECT(test::throws([&] { read_onnx("einsum_multiple_arrows_negative_test.onnx"); }));
}

TEST_CASE(einsum_empty_term_before_arrow_negative_test)
{
    EXPECT(test::throws([&] { read_onnx("einsum_empty_term_before_arrow_negative_test.onnx"); }));
}

TEST_CASE(einsum_multiple_ellipses_negative_test)
{
    EXPECT(test::throws([&] { read_onnx("einsum_multiple_ellipses_negative_test.onnx"); }));
}

TEST_CASE(einsum_comma_in_output_negative_test)
{
    EXPECT(test::throws([&] { read_onnx("einsum_comma_in_output_negative_test.onnx"); }));
}

TEST_CASE(einsum_empty_term_before_comma_negative_test)
{
    EXPECT(test::throws([&] { read_onnx("einsum_empty_term_before_comma_negative_test.onnx"); }));
}

TEST_CASE(einsum_last_input_missing_negative_test)
{
    EXPECT(test::throws([&] { read_onnx("einsum_last_input_missing_negative_test.onnx"); }));
}

TEST_CASE(einsum_term_input_mismatch_negative_test)
{
    EXPECT(test::throws([&] { read_onnx("einsum_term_input_mismatch_negative_test.onnx"); }));
}

TEST_CASE(einsum_ellipsis_mismatch_negative_test)
{
    EXPECT(test::throws([&] { read_onnx("einsum_ellipsis_mismatch_negative_test.onnx"); }));
}

TEST_CASE(einsum_rank_mismatch_negative_test)
{
    EXPECT(test::throws([&] { read_onnx("einsum_rank_mismatch_negative_test.onnx"); }));
}

TEST_CASE(einsum_output_surplus_label_negative_test)
{
    EXPECT(test::throws([&] { read_onnx("einsum_output_surplus_label_negative_test.onnx"); }));
}

TEST_CASE(einsum_output_missing_ellipsis_negative_test)
{
    EXPECT(test::throws([&] { read_onnx("einsum_output_missing_ellipsis_negative_test.onnx"); }));
}

TEST_CASE(einsum_multiple_diagonals_negative_test)
{
    EXPECT(test::throws([&] { read_onnx("einsum_multiple_diagonals_negative_test.onnx"); }));
}

TEST_CASE(einsum_diagonal_dim_mismatch_negative_test)
{
    EXPECT(test::throws([&] { read_onnx("einsum_diagonal_dim_mismatch_negative_test.onnx"); }));
}

TEST_CASE(einsum_right_batch_diagonal_negative_test)
{
    EXPECT(test::throws([&] { read_onnx("einsum_right_batch_diagonal_negative_test.onnx"); }));
}
