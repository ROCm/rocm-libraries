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

TEST_CASE(qlinearconv_test)
{
    // https://xadupre.github.io/draft/onnx/onnx_doc_folder/onnx__QLinearConv.html
    migraphx::program p = read_onnx("qlinearconv_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape sx{migraphx::shape::uint8_type, {1, 1, 7, 7}};
    std::vector<uint8_t> x_data = {255, 174, 162, 25,  203, 168, 58,  15,  59,  237, 95,  129, 0,
                                   64,  56,  242, 153, 221, 168, 12,  166, 232, 178, 186, 195, 237,
                                   162, 237, 188, 39,  124, 77,  80,  102, 43,  127, 230, 21,  83,
                                   41,  40,  134, 255, 154, 92,  141, 42,  148, 247};

    migraphx::parameter_map pp;
    pp["X"]     = migraphx::argument(sx, x_data.data());
    auto result = p.eval(pp).back();

    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<uint8_t> gold = {0,   81,  93,  230, 52,  87,  197, 240, 196, 18,  160, 126, 255,
                                 191, 199, 13,  102, 34,  87,  243, 89,  23,  77,  69,  60,  18,
                                 93,  18,  67,  216, 131, 178, 175, 153, 212, 128, 25,  234, 172,
                                 214, 215, 121, 0,   101, 163, 114, 213, 107, 8};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
