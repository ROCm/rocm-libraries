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

TEST_CASE(multi_head_attention_invalid_attribute_test)
{
    EXPECT(test::throws([&] { read_onnx("mha_invalid_attribute_test.onnx"); }));
}

TEST_CASE(multi_head_attention_invalid_input_test)
{
    EXPECT(test::throws([&] { read_onnx("mha_invalid_input_test.onnx"); }));
}

TEST_CASE(multi_head_attention_invalid_query_test)
{
    EXPECT(test::throws([&] { read_onnx("mha_invalid_query_test.onnx"); }));
}

TEST_CASE(multi_head_attention_invalid_qkv_test)
{
    EXPECT(test::throws([&] { read_onnx("mha_invalid_qkv_test.onnx"); }));
}

TEST_CASE(multi_head_attention_invalid_key_missing_test)
{
    EXPECT(test::throws([&] { read_onnx("mha_invalid_key_missing_test.onnx"); }));
}

TEST_CASE(multi_head_attention_invalid_key_ndim_test)
{
    EXPECT(test::throws([&] { read_onnx("mha_invalid_key_ndim_test.onnx"); }));
}

TEST_CASE(multi_head_attention_invalid_kv_test)
{
    EXPECT(test::throws([&] { read_onnx("mha_invalid_kv_test.onnx"); }));
}

TEST_CASE(multi_head_attention_invalid_key_test)
{
    EXPECT(test::throws([&] { read_onnx("mha_invalid_key_test.onnx"); }));
}

TEST_CASE(multi_head_attention_invalid_value_missing_test)
{
    EXPECT(test::throws([&] { read_onnx("mha_invalid_value_missing_test.onnx"); }));
}

TEST_CASE(multi_head_attention_invalid_value_test)
{
    EXPECT(test::throws([&] { read_onnx("mha_invalid_value_test.onnx"); }));
}

TEST_CASE(multi_head_attention_invalid_value_ndim_test)
{
    EXPECT(test::throws([&] { read_onnx("mha_invalid_value_ndim_test.onnx"); }));
}

TEST_CASE(multi_head_attention_invalid_cross_key_test)
{
    EXPECT(test::throws([&] { read_onnx("mha_invalid_cross_key_test.onnx"); }));
}

TEST_CASE(multi_head_attention_invalid_cross_value_test)
{
    EXPECT(test::throws([&] { read_onnx("mha_invalid_cross_value_test.onnx"); }));
}
