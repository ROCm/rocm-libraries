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

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(qlinearconv_pad_1_test)
{
    // https:xadupre.github.io/draft/onnx/onnx_doc_folder/onnx__Conv.html
    migraphx::program p = read_onnx("qlinearconv_pad_1_test.onnx");

    p.compile(migraphx::make_target("ref"));

    migraphx::shape sx{migraphx::shape::uint8_type, {1, 1, 5, 5}};

    std::vector<uint8_t> x_data = {0,   11,  21,  32,  42,  53,  64,  74,  85,  96,  106, 117, 128,
                                   138, 149, 159, 170, 181, 191, 202, 212, 223, 234, 244, 255};

    migraphx::parameter_map pp;
    pp["X"]     = migraphx::argument(sx, x_data.data());
    auto result = p.eval(pp).back();

    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // # (1, 1, 5, 5) output tensor
    std::vector<uint8_t> gold = {19,  33,  43,  52,  38,  52,  85,  99,  113, 80,  99,  156, 170,
                                 184, 128, 146, 227, 241, 255, 175, 113, 175, 184, 194, 132};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
