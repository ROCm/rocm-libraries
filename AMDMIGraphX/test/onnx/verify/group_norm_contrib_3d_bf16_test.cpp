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

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>
#include <onnx_verify_utils.hpp>

TEST_CASE(group_norm_contrib_bf16_test)
{
    using migraphx::bf16;
    std::vector<bf16> gamma{bf16{1.2}, bf16{0.8}};
    std::vector<bf16> beta{bf16{0.5}, bf16{0.2}};
    std::vector<bf16> result_vector =
        norm_test<bf16>({1, 4, 2},
                        gamma,
                        beta,
                        read_onnx("group_norm_contrib_3d_channel_last_bf16_test.onnx"),
                        "gamma",
                        "beta");
    std::vector<bf16> gold = {bf16{-1.10996256},
                              bf16{-0.87330837},
                              bf16{-0.0366542},
                              bf16{-0.15776947},
                              bf16{1.0366542},
                              bf16{0.55776947},
                              bf16{2.10996256},
                              bf16{1.27330837}};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
