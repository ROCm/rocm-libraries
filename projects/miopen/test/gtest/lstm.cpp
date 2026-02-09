/*******************************************************************************
 *
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
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

struct GPU_LSTM_FP32 : LSTM_test<float>, testing::TestWithParam<std::tuple<int, int, int, int, int>>
{
};

TEST_P(GPU_LSTM_FP32, FloatTest)
{
    int device_count{0};
    if((hipGetDeviceCount(&device_count) != hipSuccess) or (device_count == 0))
    {
        GTEST_SKIP() << "No HIP devices available for testing";
    }

    int batchSize{17};
    int seqLength{2};

    auto [usePadding, inputMode, biasMode, dirMode, algoMode] = GetParam();

    this->batchSize  = batchSize;
    this->seqLength  = seqLength;
    this->inVecLen   = batchSize;
    this->hiddenSize = 67;
    this->numLayers  = 1;
    this->batchSeq   = generate_batchSeq(batchSize, seqLength)[0];
    this->usePadding = usePadding;
    this->inputMode  = inputMode;
    this->biasMode   = biasMode;
    this->dirMode    = dirMode;
    this->algoMode   = algoMode;

    RunTest();
}

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_LSTM_FP32,
                         testing::Combine(testing::Values(0, 1),
                                          testing::Values(0, 1),
                                          testing::Values(0, 1),
                                          testing::Values(0, 1),
                                          testing::Values(0, 1)));
