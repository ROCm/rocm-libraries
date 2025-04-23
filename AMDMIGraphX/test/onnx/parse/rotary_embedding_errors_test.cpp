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

#include <onnx_test.hpp>

TEST_CASE(rotary_embedding_packed_batching_test)
{
    EXPECT(test::throws([&] { read_onnx("rotary_embedding_packed_batching_test.onnx"); }));
}

TEST_CASE(rotary_embedding_scale_test)
{
    EXPECT(test::throws([&] { read_onnx("rotary_embedding_scale_test.onnx"); }));
}

TEST_CASE(rotary_embedding_num_heads_test)
{
    EXPECT(test::throws([&] { read_onnx("rotary_embedding_num_heads_test.onnx"); }));
}

TEST_CASE(rotary_embedding_input_dims_test)
{
    EXPECT(test::throws([&] { read_onnx("rotary_embedding_input_dims_test.onnx"); }));
}

TEST_CASE(rotary_embedding_pos_ids_1_test)
{
    EXPECT(test::throws([&] { read_onnx("rotary_embedding_pos_ids_1_test.onnx"); }));
}

TEST_CASE(rotary_embedding_pos_ids_2_test)
{
    EXPECT(test::throws([&] { read_onnx("rotary_embedding_pos_ids_2_test.onnx"); }));
}

TEST_CASE(rotary_embedding_pos_ids_3_test)
{
    EXPECT(test::throws([&] { read_onnx("rotary_embedding_pos_ids_3_test.onnx"); }));
}

TEST_CASE(rotary_embedding_dim_size_test)
{
    EXPECT(test::throws([&] { read_onnx("rotary_embedding_dim_size_test.onnx"); }));
}

TEST_CASE(rotary_embedding_cache_1_test)
{
    EXPECT(test::throws([&] { read_onnx("rotary_embedding_cache_1_test.onnx"); }));
}

TEST_CASE(rotary_embedding_cache_2_test)
{
    EXPECT(test::throws([&] { read_onnx("rotary_embedding_cache_2_test.onnx"); }));
}

TEST_CASE(rotary_embedding_wrong_n_inputs_test)
{
    EXPECT(test::throws([&] { read_onnx("rotary_embedding_wrong_n_inputs_test.onnx"); }));
}
