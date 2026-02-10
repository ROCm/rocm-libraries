/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include "lstm.hpp"
#include <hip/hip_runtime.h>

struct GPU_LSTM_extra_FP32
    : LSTM_test<float>,
      testing::TestWithParam<std::tuple<int, int, int, int, int, int, int, int, int>>
{
};

TEST_P(GPU_LSTM_extra_FP32, FloatTest)
{
    int device_count{0};
    if((hipGetDeviceCount(&device_count) != hipSuccess) or (device_count == 0))
    {
        GTEST_SKIP() << "No HIP devices available for testing";
    }

    this->batchSize  = 32;
    this->seqLength  = 3;
    this->batchSeq   = {32, 32, 32};
    this->inVecLen   = 128;
    this->hiddenSize = 128;
    this->numLayers  = 1;
    this->inputMode  = 0;
    this->biasMode   = 0;

    auto [dirMode, nohx, nodhy, nocx, nodcy, nohy, nodhx, nocy, nodcx] = GetParam();

    this->dirMode = dirMode;
    this->nohx    = bool(nohx);
    this->nodhy   = bool(nodhy);
    this->nocx    = bool(nocx);
    this->nodcy   = bool(nodcy);
    this->nohy    = bool(nohy);
    this->nodhx   = bool(nodhx);
    this->nocy    = bool(nocy);
    this->nodcx   = bool(nodcx);

    RunTest();
};

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_LSTM_extra_FP32,
    testing::Values(// dir-mode no-hx no-dhy no-cx no-dcy no-hy no-dhx no-cy no-dcx
        std::make_tuple(0,       1,    0,     0,    0,     0,    0,     0,    0),
        std::make_tuple(0,       0,    1,     0,    0,     0,    0,     0,    0),
        std::make_tuple(0,       1,    1,     0,    0,     0,    0,     0,    0),
        std::make_tuple(0,       0,    0,     1,    0,     0,    0,     0,    0),
        std::make_tuple(0,       1,    0,     1,    0,     0,    0,     0,    0),
        std::make_tuple(0,       0,    0,     0,    1,     0,    0,     0,    0),
        std::make_tuple(0,       0,    0,     1,    1,     0,    0,     0,    0),
        std::make_tuple(1,       1,    0,     0,    0,     0,    0,     0,    0),
        std::make_tuple(1,       0,    1,     0,    0,     0,    0,     0,    0),
        std::make_tuple(1,       1,    1,     0,    0,     0,    0,     0,    0),
        std::make_tuple(1,       0,    0,     1,    0,     0,    0,     0,    0),
        std::make_tuple(1,       1,    0,     1,    0,     0,    0,     0,    0),
        std::make_tuple(1,       0,    0,     0,    1,     0,    0,     0,    0),
        std::make_tuple(1,       0,    0,     1,    1,     0,    0,     0,    0),
        std::make_tuple(0,       0,    0,     0,    0,     1,    0,     0,    0),
        std::make_tuple(0,       0,    0,     0,    0,     0,    1,     0,    0),
        std::make_tuple(0,       0,    0,     0,    0,     1,    1,     0,    0),
        std::make_tuple(0,       0,    0,     0,    0,     0,    0,     1,    0),
        std::make_tuple(0,       0,    0,     0,    0,     1,    0,     1,    0),
        std::make_tuple(0,       0,    0,     0,    0,     0,    0,     0,    1),
        std::make_tuple(0,       0,    0,     0,    0,     0,    0,     1,    1),
        std::make_tuple(1,       0,    0,     0,    0,     1,    0,     0,    0),
        std::make_tuple(1,       0,    0,     0,    0,     0,    1,     0,    0),
        std::make_tuple(1,       0,    0,     0,    0,     1,    1,     0,    1),
        std::make_tuple(1,       0,    0,     0,    0,     0,    0,     1,    0),
        std::make_tuple(1,       0,    0,     0,    0,     1,    0,     1,    0),
        std::make_tuple(1,       0,    0,     0,    0,     0,    0,     0,    1),
        std::make_tuple(1,       0,    0,     0,    0,     0,    0,     1,    1),
        std::make_tuple(0,       1,    1,     1,    1,     1,    1,     1,    1),
        std::make_tuple(1,       1,    1,     1,    1,     1,    1,     1,    1)));
// clang-format on
