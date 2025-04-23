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

TEST_CASE(qlinearaveragepool_2d_same_lower_test)
{
    auto p = read_onnx("qlinearaveragepool_2d_same_lower_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<uint8_t> data_x = {195, 102, 250, 61,  222, 6,   243, 218, 230, 105, 36,  116,
                                   194, 31,  113, 85,  126, 204, 80,  38,  115, 167, 221, 67,
                                   69,  140, 11,  209, 136, 120, 39,  96,  29,  5,   167, 40,
                                   58,  51,  157, 179, 244, 149, 76,  243, 126, 144, 192, 199};
    migraphx::shape s_x{migraphx::shape::uint8_type, {1, 3, 4, 4}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<uint8_t> gold = {195, 148, 176, 156, 208, 131, 150, 193, 226, 141, 98,  153,
                                 212, 140, 71,  88,  126, 165, 142, 59,  120, 153, 168, 102,
                                 92,  123, 135, 127, 102, 116, 78,  89,  29,  17,  86,  104,
                                 44,  36,  95,  136, 151, 126, 108, 164, 185, 166, 140, 178};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
