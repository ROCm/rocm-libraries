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

TEST_CASE(softmaxcrossentropyloss_label_wrong_type_test)
{
    EXPECT(test::throws(
        [&] { migraphx::parse_onnx("softmaxcrossentropyloss_label_wrong_type_test.onnx"); }));
}

TEST_CASE(softmaxcrossentropyloss_score_dim_err_test)
{
    EXPECT(test::throws(
        [&] { migraphx::parse_onnx("softmaxcrossentropyloss_score_dim_err_test.onnx"); }));
}

TEST_CASE(softmaxcrossentropyloss_score_label_mismatch_test)
{
    EXPECT(test::throws(
        [&] { migraphx::parse_onnx("softmaxcrossentropyloss_score_label_mismatch_test.onnx"); }));
}

TEST_CASE(softmaxcrossentropyloss_score_label_wrong_k_dims_test)
{
    EXPECT(test::throws([&] {
        migraphx::parse_onnx("softmaxcrossentropyloss_score_label_wrong_k_dims_test.onnx");
    }));
}

TEST_CASE(softmaxcrossentropyloss_scores_wrong_type_test)
{
    EXPECT(test::throws(
        [&] { migraphx::parse_onnx("softmaxcrossentropyloss_scores_wrong_type.onnx"); }));
}

TEST_CASE(softmaxcrossentropyloss_weight_score_mismatch_valid_type_test)
{
    EXPECT(test::throws([&] {
        migraphx::parse_onnx("softmaxcrossentropyloss_weight_score_mismatch_valid_type_test.onnx");
    }));
}

TEST_CASE(softmaxcrossentropyloss_weight_wrong_dims_test)
{
    EXPECT(test::throws(
        [&] { migraphx::parse_onnx("softmaxcrossentropyloss_weight_wrong_dims_test.onnx"); }));
}

TEST_CASE(softmaxcrossentropyloss_weight_wrong_type_test)
{
    EXPECT(test::throws(
        [&] { migraphx::parse_onnx("softmaxcrossentropyloss_weight_wrong_type_test.onnx"); }));
}

TEST_CASE(softmaxcrossentropyloss_invalid_reduction_test)
{
    EXPECT(test::throws(
        [&] { migraphx::parse_onnx("softmaxcrossentropyloss_invalid_reduction_test.onnx"); }));
}

TEST_CASE(softmaxcrossentropyloss_invalid_k_dimensions_test)
{
    EXPECT(test::throws(
        [&] { migraphx::parse_onnx("softmaxcrossentropyloss_kdim_not_equal_test.onnx"); }));
}
