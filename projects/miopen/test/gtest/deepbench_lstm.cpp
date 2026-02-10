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

struct GPU_DeepBench_LSTM_FP32 : LSTM_test<float>,
                                 testing::TestWithParam<std::tuple<int, int, int, int>>
{
};

TEST_P(GPU_DeepBench_LSTM_FP32, FloatTest)
{
    int device_count{0};
    if((hipGetDeviceCount(&device_count) != hipSuccess) or (device_count == 0))
    {
        // GTEST_SKIP() << "No HIP devices available for testing";
    }

    this->numLayers     = 1;
    this->inputMode     = 1;
    this->biasMode      = 0;
    this->dirMode       = 0;
    this->flatBatchFill = 1;

    auto [batchSize, seqLength, inVecLen, hiddenSize] = GetParam();

    this->batchSize  = batchSize;
    this->seqLength  = seqLength;
    this->inVecLen   = inVecLen;
    this->hiddenSize = hiddenSize;
};

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_DeepBench_LSTM_FP32,
    testing::Values(//batch-size seq-len vector-len hidden-size
        std::make_tuple(16,       25,      512,       512),
        std::make_tuple(32,       25,      512,       512),
        std::make_tuple(64,       25,      512,       512),
        std::make_tuple(128,      25,      512,       512),
        std::make_tuple(16,       25,      1024,      1024),
        std::make_tuple(32,       25,      1024,      1024),
        std::make_tuple(64,       25,      1024,      1024),
        std::make_tuple(128,      25,      1024,      1024),
        std::make_tuple(16,       25,      2048,      2048),
        std::make_tuple(32,       25,      2048,      2048),
        std::make_tuple(64,       25,      2048,      2048),
        std::make_tuple(128,      25,      2048,      2048),
        std::make_tuple(16,       25,      4096,      4096),
        std::make_tuple(32,       25,      4096,      4096),
        std::make_tuple(64,       25,      4096,      4096),
        std::make_tuple(128,      25,      4096,      4096),
        std::make_tuple(8,        50,      1536,      1536),
        std::make_tuple(16,       50,      1536,      1536),
        std::make_tuple(32,       50,      1536,      1536),
        std::make_tuple(16,       150,     256,       256),
        std::make_tuple(32,       150,     256,       256),
        std::make_tuple(64,       150,     256,       256)));
// clang-format on
