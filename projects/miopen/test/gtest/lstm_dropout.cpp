/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include "rnn_util.hpp"
#include <hip/hip_runtime.h>
#include <gtest/gtest_common.hpp>

struct Parameters
{
    int batchSize;
    int seqLength;
    int inVecLen;
    int hiddenSize;
    int numLayers;
    int nohx;
    int nodhy;
    int nocx;
    int nodcy;
    int nohy;
    int nodhx;
    int nocy;
    int nodcx;
    int flatBatchFill;
    int useDropout;
    int inputMode;
    int biasMode;
    int dirMode;
    int algoMode;
    std::vector<int> batchSeq;
};

auto GetTestCases(miopenDataType_t dataType)
{
    Parameters params;
    params.batchSize = 17;
    params.seqLength = 25;
    params.inVecLen = 17;
    params.hiddenSize = 67;
    params.numLayers = 3;
    params.nohx = 0;
    params.nodhy = 0;
    params.nocx = 0;
    params.nodcy = 0;
    params.nohy = 0;
    params.nodhx = 0;
    params.nocy = 0;
    params.nodcx = 0;
    params.flatBatchFill = 1;
    params.useDropout = 1;
    params.inputMode = 0;
    params.biasMode = 0;
    params.dirMode = 0;
    params.algoMode = 0;
    params.batchSeq = generate_batchSeq(params.batchSize, params.seqLength)[0];
    return params;
}

template <typename T>
struct GPU_lstm_dropout_Test : public ::testing::TestWithParam<Parameters>
{
    int device_count{0};
    miopenHandle_t handle;

    void SetUp() override
    {
        prng::reset_seed();
        miopenCreate(&handle);
        //if(hipGetDeviceCount(&device_count) != hipSuccess)
        //    device_count = 0;
    }

    void TearDown() override { miopenDestroy(handle); }

    void Run()
    {
        auto params = GetParam();
    }
};

using GPU_lstm_dropout_FP32 = GPU_lstm_dropout_Test<float>;

TEST_P(GPU_lstm_dropout_FP32, FloatTest_lstm_dropout)
{
    /*if(device_count == 0)
    {
        GTEST_SKIP() << "No HIP devices available for testing";
    }*/
    Run();
}

INSTANTIATE_TEST_SUITE_P(Full, GPU_lstm_dropout_FP32, testing::Values(GetTestCases(miopenFloat)));
