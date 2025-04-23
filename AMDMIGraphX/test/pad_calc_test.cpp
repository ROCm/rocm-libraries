/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/program.hpp>
#include <migraphx/pad_calc.hpp>
#include "test.hpp"

TEST_CASE(pad_calc_test_no_pad)
{
    std::vector<int64_t> golden_pads{0, 0, 0, 0};
    std::vector<int64_t> pads{0, 0, 0, 0};
    // 1x1 filter size
    migraphx::calculate_padding(0, pads, 16, 1, 1, 1);
    migraphx::calculate_padding(1, pads, 16, 1, 1, 1);
    EXPECT(pads == golden_pads);
}

TEST_CASE(pad_calc_test_pad_by_1)
{
    std::vector<int64_t> golden_pads{1, 1, 1, 1};
    std::vector<int64_t> pads{0, 0, 0, 0};
    // 3x3 filter size
    migraphx::calculate_padding(0, pads, 16, 1, 1, 3);
    migraphx::calculate_padding(1, pads, 16, 1, 1, 3);
    EXPECT(pads == golden_pads);
}

TEST_CASE(pad_calc_test_pad_by_1_asym_2x2_filter)
{
    std::vector<int64_t> golden_pads{0, 0, 1, 1};
    std::vector<int64_t> pads{0, 0, 0, 0};
    // 2x2 filter size
    migraphx::calculate_padding(0, pads, 16, 1, 1, 2);
    migraphx::calculate_padding(1, pads, 16, 1, 1, 2);
    EXPECT(pads == golden_pads);
}

TEST_CASE(pad_calc_test_pad_by_2)
{
    std::vector<int64_t> golden_pads{2, 2, 2, 2};
    std::vector<int64_t> pads{0, 0, 0, 0};
    // 5x5 filter size
    migraphx::calculate_padding(0, pads, 16, 1, 1, 5);
    migraphx::calculate_padding(1, pads, 16, 1, 1, 5);
    EXPECT(pads == golden_pads);
}

TEST_CASE(pad_calc_test_pad_by_1_asym_stride_2)
{
    std::vector<int64_t> golden_pads{0, 0, 1, 1};
    std::vector<int64_t> pads{0, 0, 0, 0};
    // 3x3 filter size
    migraphx::calculate_padding(0, pads, 16, 2, 1, 3);
    migraphx::calculate_padding(1, pads, 16, 2, 1, 3);
    EXPECT(pads == golden_pads);
}

TEST_CASE(pad_calc_test_dilation_2)
{
    std::vector<int64_t> golden_pads{2, 2, 2, 2};
    std::vector<int64_t> pads{0, 0, 0, 0};
    // 3x3 filter size
    migraphx::calculate_padding(0, pads, 16, 1, 2, 3);
    migraphx::calculate_padding(1, pads, 16, 1, 2, 3);
    EXPECT(pads == golden_pads);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
