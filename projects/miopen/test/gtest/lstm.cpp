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

namespace {

struct TestCase
{
    TestCase(int usePadding_, int inputMode_, int biasMode_, int dirMode_, int algoMode_)
        : usePadding(usePadding_),
          inputMode(inputMode_),
          biasMode(biasMode_),
          dirMode(dirMode_),
          algoMode(algoMode_)
    {
    }

    int usePadding;
    int inputMode;
    int biasMode;
    int dirMode;
    int algoMode;

    friend std::ostream& operator<<(std::ostream& os, const TestCase& tc)
    {
        return os << "use-padding:" << tc.usePadding << " input-mode:" << tc.inputMode
                  << " bias-mode:" << tc.biasMode << " dir-mode:" << tc.dirMode
                  << " algo-mode:" << tc.algoMode;
    }
};

std::vector<TestCase> GetTestCases()
{
    return std::vector{
        // clang-format off
        //  use-padding input-mode, bias-mode, dir-mode, algo-mode
        TestCase(0,        0,         0,         0,        0),
        TestCase(0,        0,         0,         0,        1),
        TestCase(0,        0,         0,         1,        0),
        TestCase(0,        0,         0,         1,        1),
        TestCase(0,        0,         1,         0,        0),
        TestCase(0,        0,         1,         0,        1),
        TestCase(0,        0,         1,         1,        0),
        TestCase(0,        0,         1,         1,        1),
        TestCase(0,        1,         0,         0,        0),
        TestCase(0,        1,         0,         0,        1),
        TestCase(0,        1,         0,         1,        0),
        TestCase(0,        1,         0,         1,        1),
        TestCase(0,        1,         1,         0,        0),
        TestCase(0,        1,         1,         0,        1),
        TestCase(0,        1,         1,         1,        0),
        TestCase(0,        1,         1,         1,        1),
        TestCase(1,        0,         0,         0,        0),
        TestCase(1,        0,         0,         0,        1),
        TestCase(1,        0,         0,         1,        0),
        TestCase(1,        0,         0,         1,        1),
        TestCase(1,        0,         1,         0,        0),
        TestCase(1,        0,         1,         0,        1),
        TestCase(1,        0,         1,         1,        0),
        TestCase(1,        0,         1,         1,        1),
        TestCase(1,        1,         0,         0,        0),
        TestCase(1,        1,         0,         0,        1),
        TestCase(1,        1,         0,         1,        0),
        TestCase(1,        1,         0,         1,        1),
        TestCase(1,        1,         1,         0,        0),
        TestCase(1,        1,         1,         0,        1),
        TestCase(1,        1,         1,         1,        0),
        TestCase(1,        1,         1,         1,        1)
        // clang-format on
    };
}

} // namespace

template <typename T>
struct GPU_LSTM_test : LSTM_test<T>, testing::TestWithParam<TestCase>
{
protected:
    void SetUp() override
    {
        this->dataType = miopen_type<T>{};

        auto [usePadding, inputMode, biasMode, dirMode, algoMode] = GetParam();

        int batchSize{17};
        int seqLength{2};
        this->batchSize  = batchSize;
        this->seqLength  = seqLength;
        this->inVecLen   = batchSize;
        this->batchSeq   = generate_batchSeq(batchSize, seqLength)[0];
        this->hiddenSize = 67;
        this->usePadding = usePadding;
        this->inputMode  = inputMode;
        this->biasMode   = biasMode;
        this->dirMode    = dirMode;
        this->algoMode   = algoMode;
    }
};

using GPU_LSTM_FP16 = GPU_LSTM_test<half_float::half>;

TEST_P(GPU_LSTM_FP16, HalfTest) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(Full, GPU_LSTM_FP16, testing::ValuesIn(GetTestCases()));

using GPU_LSTM_FP32 = GPU_LSTM_test<float>;

TEST_P(GPU_LSTM_FP32, FloatTest) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(Full, GPU_LSTM_FP32, testing::ValuesIn(GetTestCases()));
