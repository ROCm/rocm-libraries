/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "unit_conv_solver.hpp"
#include "get_handle.hpp"

namespace {

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    auto cases = std::vector{
        // clang-format off
        TestCase{{2, 8, 8, 8}, {8, 8, 3, 3}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{1, 4, 5, 5}, {4, 4, 3, 3}, {1, 1}, {1, 1}, {1, 1}, datatype}, // Padding
        TestCase{{1, 3, 7, 7}, {3, 3, 3, 3}, {0, 0}, {2, 2}, {1, 1}, datatype}, // Strides
        TestCase{{1, 3, 7, 7}, {3, 3, 3, 3}, {0, 0}, {3, 3}, {1, 1}, datatype}, // large stride
        TestCase{{1, 3, 7, 7}, {3, 3, 3, 3}, {0, 0}, {1, 1}, {2, 2}, datatype}, // Dilation
        TestCase{{1, 3, 7, 7}, {3, 3, 3, 3}, {1, 1}, {1, 1}, {3, 3}, datatype}, // high dilation
        TestCase{{1, 1, 4, 4}, {1, 1, 5, 5}, {0, 0}, {1, 1}, {1, 1}, datatype}, // kernel larger than input
        TestCase{{1, 1, 4, 4}, {1, 1, 4, 4}, {0, 0}, {1, 1}, {1, 1}, datatype}, // kernel equal to input
        TestCase{{1, 8, 10, 5}, {8, 8, 3, 3}, {1, 1}, {1, 1}, {1, 1}, datatype}, // Non-square input
        TestCase{{1, 8, 8, 8}, {8, 8, 3, 5}, {1, 2}, {1, 1}, {1, 1}, datatype}, // Non-square kernel
        TestCase{{4, 8, 8, 8}, {8, 8, 3, 3}, {1, 1}, {1, 1}, {1, 1}, datatype}, // Batch size > 1
        TestCase{{1, 3, 7, 7}, {3, 3, 3, 3}, {1, 1}, {1, 1}, {2, 3}, datatype}, // Uneven dilation
        TestCase{{2, 32, 64, 64}, {32, 32, 5, 5}, {2, 2}, {1, 1}, {1, 1}, datatype}, // Large input size
        TestCase{{1, 3, 15, 15}, {3, 3, 3, 3}, {2, 2}, {2, 2}, {3, 3}, datatype}, // High dilation + stride + padding
        TestCase{{1, 7, 16, 16}, {7, 7, 3, 3}, {1, 1}, {1, 1}, {1, 1}, datatype}, // Unusual channel counts
        TestCase{{1, 5, 16, 16}, {5, 5, 3, 3}, {1, 1}, {1, 1}, {1, 1}, datatype},
        TestCase{{1, 4, 5, 5}, {4, 4, 3, 3}, {3, 3}, {1, 1}, {1, 1}, datatype}, // large padding
        TestCase{{1, 3, 7, 7}, {3, 3, 3, 3}, {1, 2}, {1, 1}, {1, 1}, datatype}, // Uneven padding (asymmetric)
        TestCase{{1, 3, 7, 7}, {3, 3, 3, 3}, {1, 1}, {2, 1}, {1, 1}, datatype}, // Uneven stride
        TestCase{{2, 8, 8, 8}, {8, 8, 3, 3}, {0, 0}, {1, 1}, {1, 1},datatype, miopenTensorNHWC},
        TestCase{{1, 3, 7, 7}, {3, 3, 3, 3}, {0, 0}, {2, 2}, {1, 1},datatype, miopenTensorNHWC}, // Strides
        TestCase{{1, 3, 7, 7}, {3, 3, 3, 3}, {0, 0}, {1, 1}, {2, 2},datatype, miopenTensorNHWC}, // Dilation
        TestCase{{1, 1, 4, 4}, {1, 1, 5, 5}, {0, 0}, {1, 1}, {1, 1},datatype, miopenTensorNHWC}, // kernel larger than input
        TestCase{{1, 1, 4, 4}, {1, 1, 4, 4}, {0, 0}, {1, 1}, {1, 1},datatype, miopenTensorNHWC}, // kernel equal to input
        TestCase{{1, 8, 10, 5}, {8, 8, 3, 3}, {1, 1}, {1, 1}, {1, 1},datatype, miopenTensorNHWC}, // Non-square input
        TestCase{{1, 8, 8, 8}, {8, 8, 3, 5}, {1, 2}, {1, 1}, {1, 1},datatype, miopenTensorNHWC}, // Non-square kernel
        TestCase{{4, 8, 8, 8}, {8, 8, 3, 3}, {1, 1}, {1, 1}, {1, 1},datatype, miopenTensorNHWC}, // Batch size > 1
        TestCase{{1, 3, 7, 7}, {3, 3, 3, 3}, {1, 1}, {1, 1}, {2, 3},datatype, miopenTensorNHWC}, // Uneven dilation
        TestCase{{2, 32, 64, 64}, {32, 32, 5, 5}, {2, 2}, {1, 1}, {1, 1},datatype, miopenTensorNHWC}, // Large input size
        TestCase{{1, 3, 15, 15}, {3, 3, 3, 3}, {2, 2}, {2, 2}, {3, 3},datatype, miopenTensorNHWC}, // High dilation + stride + padding
        TestCase{{1, 7, 16, 16}, {7, 7, 3, 3}, {1, 1}, {1, 1}, {1, 1},datatype, miopenTensorNHWC}, // Unusual channel counts
        TestCase{{1, 5, 16, 16}, {5, 5, 3, 3}, {1, 1}, {1, 1}, {1, 1},datatype, miopenTensorNHWC},
        TestCase{{1, 128, 8, 8}, {128, 128, 3, 3}, {1, 1}, {1, 1}, {1, 1},datatype, miopenTensorNHWC}, // High channel count (stress test inner dim in NHWC)
        TestCase{{16, 16, 8, 8}, {16, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1},datatype, miopenTensorNHWC}, // Large batch
        TestCase{{1, 4, 5, 5}, {4, 4, 3, 3}, {1, 1}, {1, 1}, {1, 1},datatype, miopenTensorNHWC}, // Padding
        TestCase{{1, 3, 7, 7}, {3, 3, 3, 3}, {0, 0}, {3, 3}, {1, 1},datatype, miopenTensorNHWC}, // large stride
        TestCase{{1, 3, 7, 7}, {3, 3, 3, 3}, {1, 1}, {1, 1}, {3, 3},datatype, miopenTensorNHWC}, // high dilation


        TestCase{{1,  4,   16,  16},  {4,   2,   3,  3},  {0,  0},  {1,  1},  {1,  1},  2  ,datatype,miopenTensorNCHW},
        TestCase{{1,  2,   3,   3},   {2,   1,   2,  2},  {0,  0},  {1,  1},  {1,  1},  2  ,datatype,miopenTensorNCHW},
        TestCase{{1,  2,   3,   3},   {2,   1,   2,  2},  {0,  0},  {1,  1},  {1,  1},  2  ,datatype,miopenTensorNCHW},
        TestCase{{1,  4,   3,   3},   {2,   2,   2,  2},  {0,  0},  {1,  1},  {1,  1},  2  ,datatype,miopenTensorNCHW},
        TestCase{{2,  4,   3,   3},   {4,   2,   2,  2},  {0,  0},  {1,  1},  {1,  1},  2  ,datatype,miopenTensorNCHW},
        TestCase{{1,  4,   3,   3},   {4,   2,   2,  2},  {0,  0},  {1,  1},  {1,  1},  2  ,datatype,miopenTensorNCHW},
        TestCase{{1,  3,   7,   7},   {12,  1,   5,  5},  {0,  0},  {1,  1},  {2,  2},  3  ,datatype,miopenTensorNCHW},
        TestCase{{1,  3,   5,   5},   {9,   1,   3,  3},  {0,  0},  {2,  2},  {1,  1},  3  ,datatype,miopenTensorNCHW},
        TestCase{{1,  3,   3,   3},   {6,   1,   3,  3},  {0,  0},  {1,  1},  {1,  1},  3  ,datatype,miopenTensorNCHW},
        TestCase{{1,  3,   2,   2},   {3,   1,   3,  3},  {1,  1},  {1,  1},  {1,  1},  3  ,datatype,miopenTensorNCHW},
        TestCase{{1,  3,   24,  24},  {3,   1,   5,  5},  {3,  3},  {2,  2},  {2,  2},  3  ,datatype,miopenTensorNCHW},
        TestCase{{2,  8,   16,  16},  {16,  2,   3,  3},  {1,  1},  {1,  1},  {1,  1},  4  ,datatype,miopenTensorNCHW},
        TestCase{{1,  16,  64,  64},  {32,  1,   5,  5},  {2,  2},  {1,  1},  {2,  2},  16 ,datatype,miopenTensorNCHW},
        TestCase{{1,  4,   14,  14},  {4,   2,   3,  3},  {2,  2},  {1,  1},  {3,  3},  2  ,datatype,miopenTensorNCHW},
        TestCase{{1,  8,   32,  32},  {16,  4,   3,  3},  {1,  1},  {4,  4},  {1,  1},  2  ,datatype,miopenTensorNCHW},
        TestCase{{1,  4,   28,  28},  {6,   2,   5,  5},  {2,  2},  {3,  3},  {2,  2},  2  ,datatype,miopenTensorNCHW},
        TestCase{{1,  4,   20,  20},  {4,   2,   3,  3},  {0,  0},  {1,  1},  {5,  5},  2  ,datatype,miopenTensorNCHW},
        TestCase{{1,  64,  2,   2},   {64,  4,   2,  2},  {0,  0},  {1,  1},  {1,  1},  16 ,datatype,miopenTensorNCHW},
        TestCase{{2,  8,   16,  16},  {16,  1,   3,  3},  {1,  1},  {1,  1},  {1,  1},  8  ,datatype,miopenTensorNCHW},
        TestCase{{1,  16,  64,  64},  {32,  2,   5,  5},  {2,  2},  {1,  1},  {2,  2},  8  ,datatype,miopenTensorNCHW},
        TestCase{{2,  8,   16,  16},  {16,  1,   3,  3},  {1,  1},  {1,  1},  {1,  1},  8  ,datatype,miopenTensorNCHW},
        TestCase{{1,  16,  64,  64},  {32,  2,   5,  5},  {2,  2},  {1,  1},  {2,  2},  8  ,datatype,miopenTensorNCHW},
        TestCase{{1,  64,  2,   2},   {64,  32,  2,  2},  {0,  0},  {1,  1},  {1,  1},  2  ,datatype,miopenTensorNCHW},
        TestCase{{1,  4,   5,   5},   {4,   2,   3,  3},  {3,  3},  {1,  1},  {1,  1},  2  ,datatype,miopenTensorNCHW},
        TestCase{{1,  4,   7,   7},   {6,   2,   3,  3},  {1,  2},  {1,  1},  {1,  1},  2  ,datatype,miopenTensorNCHW},
        TestCase{{1,  4,   7,   7},   {8,   2,   3,  3},  {1,  1},  {2,  1},  {1,  1},  2  ,datatype,miopenTensorNCHW},
        TestCase{{1,  4,   5,   5},   {4,   2,   3,  3},  {0,  0},  {1,  1},  {1,  1},  2  ,datatype,miopenTensorNCHW},
        TestCase{{1,  4,   16,  16},  {4,   2,   3,  3},  {0,  0},  {1,  1},  {1,  1},  2  ,datatype,miopenTensorNHWC},
        TestCase{{1,  2,   3,   3},   {2,   1,   2,  2},  {0,  0},  {1,  1},  {1,  1},  2  ,datatype,miopenTensorNHWC},
        TestCase{{1,  2,   3,   3},   {2,   1,   2,  2},  {0,  0},  {1,  1},  {1,  1},  2  ,datatype,miopenTensorNHWC},
        TestCase{{1,  4,   3,   3},   {2,   2,   2,  2},  {0,  0},  {1,  1},  {1,  1},  2  ,datatype,miopenTensorNHWC},
        TestCase{{2,  4,   3,   3},   {4,   2,   2,  2},  {0,  0},  {1,  1},  {1,  1},  2  ,datatype,miopenTensorNHWC},
        TestCase{{1,  4,   3,   3},   {4,   2,   2,  2},  {0,  0},  {1,  1},  {1,  1},  2  ,datatype,miopenTensorNHWC},
        TestCase{{1,  3,   7,   7},   {12,  1,   5,  5},  {0,  0},  {1,  1},  {2,  2},  3  ,datatype,miopenTensorNHWC},
        TestCase{{1,  3,   5,   5},   {9,   1,   3,  3},  {0,  0},  {2,  2},  {1,  1},  3  ,datatype,miopenTensorNHWC},
        TestCase{{1,  3,   3,   3},   {6,   1,   3,  3},  {0,  0},  {1,  1},  {1,  1},  3  ,datatype,miopenTensorNHWC},
        TestCase{{1,  3,   2,   2},   {3,   1,   3,  3},  {1,  1},  {1,  1},  {1,  1},  3  ,datatype,miopenTensorNHWC},
        TestCase{{1,  3,   24,  24},  {3,   1,   5,  5},  {3,  3},  {2,  2},  {2,  2},  3  ,datatype,miopenTensorNHWC},
        TestCase{{1,  16,  64,  64},  {32,  1,   5,  5},  {2,  2},  {1,  1},  {2,  2},  16 ,datatype,miopenTensorNHWC},
        TestCase{{2,  8,   16,  16},  {16,  2,   3,  3},  {1,  1},  {1,  1},  {1,  1},  4  ,datatype,miopenTensorNHWC},
        TestCase{{2,  8,   16,  16},  {16,  1,   3,  3},  {1,  1},  {1,  1},  {1,  1},  8  ,datatype,miopenTensorNHWC},
        TestCase{{1,  16,  64,  64},  {32,  2,   5,  5},  {2,  2},  {1,  1},  {2,  2},  8  ,datatype,miopenTensorNHWC},
        TestCase{{1,  64,  2,   2},   {64,  4,   2,  2},  {0,  0},  {1,  1},  {1,  1},  16 ,datatype,miopenTensorNHWC},
        TestCase{{1,  64,  2,   2},   {64,  32,  2,  2},  {0,  0},  {1,  1},  {1,  1},  2  ,datatype,miopenTensorNHWC},
        TestCase{{1,  4,   14,  14},  {4,   2,   3,  3},  {2,  2},  {1,  1},  {3,  3},  2  ,datatype,miopenTensorNHWC},
        TestCase{{1,  8,   32,  32},  {16,  4,   3,  3},  {1,  1},  {4,  4},  {1,  1},  2  ,datatype,miopenTensorNHWC},
        TestCase{{1,  4,   28,  28},  {6,   2,   5,  5},  {2,  2},  {3,  3},  {2,  2},  2  ,datatype,miopenTensorNHWC},
        TestCase{{1,  4,   20,  20},  {4,   2,   3,  3},  {0,  0},  {1,  1},  {5,  5},  2  ,datatype,miopenTensorNHWC},
        TestCase{{1,  4,   5,   5},   {4,   2,   3,  3},  {3,  3},  {1,  1},  {1,  1},  2  ,datatype,miopenTensorNHWC},
        TestCase{{1,  4,   7,   7},   {6,   2,   3,  3},  {1,  2},  {1,  1},  {1,  1},  2  ,datatype,miopenTensorNHWC},
        TestCase{{1,  4,   7,   7},   {8,   2,   3,  3},  {1,  1},  {2,  1},  {1,  1},  2  ,datatype,miopenTensorNHWC},
        TestCase{{1,  4,   5,   5},   {4,   2,   3,  3},  {0,  0},  {1,  1},  {1,  1},  2  ,datatype,miopenTensorNHWC},

        TestCase{{2, 16, 5, 5, 5}, {32, 16, 1, 1, 1}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, datatype, datatype, datatype, miopenTensorNDHWC, miopenTensorNDHWC},
        TestCase{{1, 64, 7, 7, 7}, {16, 64, 1, 1, 1}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, datatype, datatype, datatype, miopenTensorNDHWC, miopenTensorNDHWC},
        TestCase{{1, 8, 14, 14, 14}, {16, 8, 3, 3, 3}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, datatype, datatype, datatype, miopenTensorNDHWC, miopenTensorNDHWC},
        TestCase{{4, 3, 10, 10, 10}, {8, 3, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, datatype, datatype, datatype, miopenTensorNDHWC, miopenTensorNDHWC},
        TestCase{{2, 4, 16, 16, 16}, {8, 4, 3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}, datatype, datatype, datatype, miopenTensorNDHWC, miopenTensorNDHWC},
        TestCase{{1, 3, 15, 30, 30}, {16, 3, 5, 7, 7}, {2, 3, 3}, {3, 4, 4}, {1, 1, 1}, datatype, datatype, datatype, miopenTensorNDHWC, miopenTensorNDHWC},
        TestCase{{1, 8, 15, 15, 15}, {8, 8, 3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {2, 2, 2}, datatype, datatype, datatype, miopenTensorNDHWC, miopenTensorNDHWC},
        TestCase{{2, 4, 20, 20, 20}, {4, 4, 3, 3, 3}, {3, 3, 3}, {1, 1, 1}, {3, 3, 3}, datatype, datatype, datatype, miopenTensorNDHWC, miopenTensorNDHWC},
        TestCase{{2, 3, 8, 16, 16}, {16, 3, 1, 3, 3}, {0, 1, 1}, {1, 1, 1}, {1, 1, 1}, datatype, datatype, datatype, miopenTensorNDHWC, miopenTensorNDHWC},
        TestCase{{1, 16, 10, 8, 8}, {16, 16, 3, 1, 1}, {1, 0, 0}, {1, 1, 1}, {1, 1, 1}, datatype, datatype, datatype, miopenTensorNDHWC, miopenTensorNDHWC},
        TestCase{{1, 4, 7, 9, 11}, {8, 4, 3, 2, 4}, {0, 1, 0}, {1, 2, 3}, {2, 1, 1}, datatype, datatype, datatype, miopenTensorNDHWC, miopenTensorNDHWC},
        TestCase{{2, 8, 3, 3, 3}, {16, 8, 3, 3, 3}, {0, 0, 0}, {3, 3, 3}, {1, 1, 1}, datatype, datatype, datatype, miopenTensorNDHWC, miopenTensorNDHWC},

        TestCase{{2, 16, 5, 5, 5}, {32, 8, 1, 1, 1}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, datatype, datatype, datatype, miopenTensorNDHWC, miopenTensorNDHWC, 2},
        TestCase{{1, 64, 7, 7, 7}, {16, 32, 1, 1, 1}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, datatype, datatype, datatype, miopenTensorNDHWC, miopenTensorNDHWC, 2},
        TestCase{{1, 8, 14, 14, 14}, {16, 4, 3, 3, 3}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, datatype, datatype, datatype, miopenTensorNDHWC, miopenTensorNDHWC, 2},
        TestCase{{1, 8, 14, 14, 14}, {16, 2, 3, 3, 3}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, datatype, datatype, datatype, miopenTensorNDHWC, miopenTensorNDHWC, 4},
        TestCase{{1, 8, 14, 14, 14}, {16, 1, 3, 3, 3}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, datatype, datatype, datatype, miopenTensorNDHWC, miopenTensorNDHWC, 8},
        TestCase{{2, 4, 16, 16, 16}, {8, 2, 3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}, datatype, datatype, datatype, miopenTensorNDHWC, miopenTensorNDHWC, 2},
        TestCase{{1, 8, 15, 15, 15}, {8, 4, 3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {2, 2, 2}, datatype, datatype, datatype, miopenTensorNDHWC, miopenTensorNDHWC, 2},
        TestCase{{2, 4, 20, 20, 20}, {4, 2, 3, 3, 3}, {3, 3, 3}, {1, 1, 1}, {3, 3, 3}, datatype, datatype, datatype, miopenTensorNDHWC, miopenTensorNDHWC, 2},
        TestCase{{1, 16, 10, 8, 8}, {16, 8, 3, 1, 1}, {1, 0, 0}, {1, 1, 1}, {1, 1, 1}, datatype, datatype, datatype, miopenTensorNDHWC, miopenTensorNDHWC, 2}
        // clang-format on
    };

    // Point-output shapes (stride == filter, output spatially 1x1) take the single-GEMM path.
    // FP32 is left out: at K=1280 its RMS error sits just under the 1.0*eps threshold that
    // non-TF32 GPUs use, so the case is not reliable there.
    if(datatype == miopenHalf)
    {
        // clang-format off
        cases.emplace_back(TestCase{{4, 3, 14, 14}, {1280, 3, 14, 14}, {0, 0}, {14, 14}, {1, 1}, datatype});
        cases.emplace_back(TestCase{{4, 3, 4, 4, 4}, {512, 3, 4, 4, 4}, {0, 0, 0}, {4, 4, 4}, {1, 1, 1}, datatype});
        cases.emplace_back(TestCase{{4, 3, 5, 5, 5}, {512, 3, 4, 4, 4}, {0, 0, 0}, {4, 4, 4}, {1, 1, 1}, datatype});
        cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {4, 4, 14, 14}},
                                    {datatype, miopenTensorNHWC, {64, 4, 14, 14}},
                                    datatype, {{0, 0}, {14, 14}, {1, 1}}});
        cases.emplace_back(TestCase{{datatype, miopenTensorNDHWC, {4, 4, 4, 4, 4}},
                                    {datatype, miopenTensorNDHWC, {64, 4, 4, 4, 4}},
                                    datatype, {{0, 0, 0}, {4, 4, 4}, {1, 1, 1}}});
        // clang-format on
    }

    return cases;
}

// Point-output bf16, covering both the direct write into dx (dx spatial equals the filter) and
// the 3D case where a larger dx has to be scattered out of the workspace by Col2Im.
auto GetConvTestCasesPointOutputBf16()
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    constexpr auto datatype = miopenBFloat16;

    return std::vector{
        // clang-format off
        TestCase{{4, 3, 14, 14}, {1280, 3, 14, 14}, {0, 0}, {14, 14}, {1, 1}, datatype},
        TestCase{{4, 3, 4, 4, 4}, {512, 3, 4, 4, 4}, {0, 0, 0}, {4, 4, 4}, {1, 1, 1}, datatype},
        TestCase{{4, 3, 5, 5, 5}, {512, 3, 4, 4, 4}, {0, 0, 0}, {4, 4, 4}, {1, 1, 1}, datatype},
        // clang-format on
    };
}

auto GetConvTestCasesFull(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    auto cases = std::vector<TestCase>{};

    if(datatype == miopenHalf)
    {
        const miopen::Handle& handle = get_handle();
        const std::string name       = handle.GetDeviceName();

        // clang-format off
        if(!miopen::StartsWith(name, "gfx1151"))
        {
            // Regression test for https://github.com/ROCm/MIOpen/issues/1956
            cases.emplace_back(TestCase{{2, 64, 128, 128, 128}, {32, 64, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenHalf});
        }
        else
        {
            //https://github.com/ROCm/TheRock/issues/3202
            // We have out of memory error on gfx1151 (flaky issue), so let's reduce tensor size for this type of machibes
            cases.emplace_back(TestCase{{2, 64, 64, 64, 64}, {32, 64, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenHalf});
        }
        // clang-format on
    }

    return cases;
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::All);
        // GEMM followed by col2im reconstruction reaches 2.59 FP16 eps RMS on gfx90a.
        p.SetTolerance(Gpu::gfx90A, miopenHalf, 3.0f);
        return p;
    }();
    return params;
}

// These shapes issue a BF16->BF16 GEMM, which rocBLAS does not support on gfx90a, so exclude it
// here; every other GPU runs the point-output bf16 cases.
// TODO: Remove this exclusion once the rocBLAS bug is fixed.
const auto& GetTestParamsNoGfx90A()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::All & ~Gpu::gfx90A);
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverGemmBwdRestBwd_FP16  = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverGemmBwdRestBwd_BFP16 = GPU_UnitTestConvSolverBwd_BFP16;
using GPU_UnitTestConvSolverGemmBwdRestBwd_FP32  = GPU_UnitTestConvSolverBwd_FP32;

using CPU_UnitTestConvSolverDevApplicabilityGemmBwdRestBwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityBwd_NONE;

TEST_P(GPU_UnitTestConvSolverGemmBwdRestBwd_FP16, GemmBwdRest)
{
    this->RunTest(miopen::solver::conv::GemmBwdRest{});
};

TEST_P(GPU_UnitTestConvSolverGemmBwdRestBwd_BFP16, GemmBwdRest)
{
    this->RunTest(miopen::solver::conv::GemmBwdRest{});
};

TEST_P(GPU_UnitTestConvSolverGemmBwdRestBwd_FP32, GemmBwdRest)
{
    this->RunTest(miopen::solver::conv::GemmBwdRest{});
};

TEST_P(CPU_UnitTestConvSolverDevApplicabilityGemmBwdRestBwd_NONE, GemmBwdRest)
{
    this->RunTest(miopen::solver::conv::GemmBwdRest{});
};

TEST(CPU_UnitTestConvSolverGemmBwdRestBwd_NONE, RejectsPackedLayout)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    const auto test_case = TestCase{
        {1, 8, 8, 8}, {8, 8, 3, 3}, {0, 0}, {1, 1}, {1, 1}, miopenFloat, miopenTensorNCHWc4};
    const auto problem = test_case.GetProblemDescription(miopen::conv::Direction::BackwardData);
    auto context       = miopen::ExecutionContext{&get_handle()};
    problem.SetupFloats(context);
    problem.SetupComputeType(context);

    EXPECT_FALSE(miopen::solver::conv::GemmBwdRest{}.IsApplicable(context, problem));
}

TEST(CPU_UnitTestConvSolverGemmBwdRestBwd_NONE, RejectsMixedPointOutputLayouts)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    const auto test_case = TestCase{{4, 3, 4, 4, 4},
                                    {16, 3, 4, 4, 4},
                                    {0, 0, 0},
                                    {4, 4, 4},
                                    {1, 1, 1},
                                    miopenHalf,
                                    miopenHalf,
                                    miopenHalf,
                                    miopenTensorNDHWC,
                                    miopenTensorNCDHW};
    const auto problem   = test_case.GetProblemDescription(miopen::conv::Direction::BackwardData);
    auto context         = miopen::ExecutionContext{&get_handle()};
    problem.SetupFloats(context);
    problem.SetupComputeType(context);

    EXPECT_FALSE(miopen::solver::conv::GemmBwdRest{}.IsApplicable(context, problem));
}

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmBwdRestBwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmBwdRestBwd_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(SmokePointOutput,
                         GPU_UnitTestConvSolverGemmBwdRestBwd_BFP16,
                         testing::Combine(testing::Values(GetTestParamsNoGfx90A()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesPointOutputBf16())));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmBwdRestBwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverDevApplicabilityGemmBwdRestBwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));

// Full tests
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverGemmBwdRestBwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenHalf))));
