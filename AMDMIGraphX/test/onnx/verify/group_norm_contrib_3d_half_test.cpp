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

TEST_CASE(group_norm_contrib_half_test)
{
    using migraphx::half;
    std::vector<half> gamma{half{1.2}, half{0.8}};
    std::vector<half> beta{half{0.5}, half{0.2}};
    std::vector<half> result_vector =
        norm_test<half>({1, 4, 2},
                        gamma,
                        beta,
                        read_onnx("group_norm_contrib_3d_channel_last_half_test.onnx"),
                        "gamma",
                        "beta");
    std::vector<half> gold = {half{-1.10996256},
                              half{-0.87330837},
                              half{-0.0366542},
                              half{-0.15776947},
                              half{1.0366542},
                              half{0.55776947},
                              half{2.10996256},
                              half{1.27330837}};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
