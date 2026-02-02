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

#include "lstm_test.hpp"
#include <hip/hip_runtime.h>
#include <gtest/gtest_common.hpp>

namespace lstm_dropout {

auto GetTestCases()
{
    int batchSize{17};
    int seqLength{25};
    int inVecLen{17};
    int hiddenSize{67};
    int numLayers{3};
    int useDropout{1};
    int usePadding{0};
    int flatBatchFill{1};
    int inputMode{0};
    int biasMode{0};
    int dirMode{0};
    int algoMode{0};
    std::vector<int> batchSeq = generate_batchSeq(batchSize, seqLength)[0];

    return std::make_tuple(batchSize,
                           seqLength,
                           inVecLen,
                           hiddenSize,
                           numLayers,
                           useDropout,
                           usePadding,
                           flatBatchFill,
                           inputMode,
                           biasMode,
                           dirMode,
                           algoMode,
                           batchSeq);
}

struct GPU_Test_FP32 : testing::TestWithParam<decltype(GetTestCases())>, LSTM_test<float>
{
    int device_count{0};

    void SetUp() override
    {
        if(hipGetDeviceCount(&device_count) != hipSuccess)
            device_count = 0;
    }
};

} // namespace lstm_dropout

using namespace lstm_dropout;

TEST_P(GPU_Test_FP32, FloatTest_lstm_dropout)
{
    if(device_count == 0)
    {
        GTEST_SKIP() << "No HIP devices available for testing";
    }

    auto [batchSize,
          seqLength,
          inVecLen,
          hiddenSize,
          numLayers,
          useDropout,
          usePadding,
          flatBatchFill,
          inputMode,
          biasMode,
          dirMode,
          algoMode,
          batchSeq] = GetParam();

    this->batchSize     = batchSize;
    this->seqLength     = seqLength;
    this->inVecLen      = inVecLen;
    this->hiddenSize    = hiddenSize;
    this->numLayers     = numLayers;
    this->useDropout    = useDropout;
    this->usePadding    = usePadding;
    this->flatBatchFill = flatBatchFill;
    this->inputMode     = inputMode;
    this->biasMode      = biasMode;
    this->dirMode       = dirMode;
    this->algoMode      = algoMode;
    this->batchSeq      = batchSeq;

    RunTest();
}

INSTANTIATE_TEST_SUITE_P(Full, GPU_Test_FP32, testing::Values(GetTestCases()));
