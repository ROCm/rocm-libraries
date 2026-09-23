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
    TestCase(int dirMode_, int flatBatchFill_) : dirMode(dirMode_), flatBatchFill(flatBatchFill_) {}

    int dirMode;
    int flatBatchFill;

    friend std::ostream& operator<<(std::ostream& os, const TestCase& tc)
    {
        return os << "dir-mode:" << tc.dirMode << " flat-batch-fill:" << tc.flatBatchFill;
    }
};

std::vector<TestCase> GetTestCases()
{
    return std::vector{
        // clang-format off
        //    dir-mode  flat-batch-fill
        TestCase(0,        0),
        TestCase(0,        1),
        TestCase(1,        0),
        TestCase(1,        1)
        // clang-format on
    };
}

} // namespace

template <typename T>
struct GPU_LSTM_dropout_test : LSTM_test<T>, testing::TestWithParam<TestCase>
{
protected:
    void SetUp() override
    {
        this->dataType = miopen_type<T>{};

        auto [dirMode, flatBatchFill] = GetParam();

        int useDropout{1};
        int batchSize{17};
        int seqLength{25};
        this->useDropout    = useDropout;
        this->batchSize     = batchSize;
        this->seqLength     = seqLength;
        this->inVecLen      = batchSize;
        this->batchSeq      = generate_batchSeq(batchSize, seqLength)[0];
        this->numLayers     = 3;
        this->hiddenSize    = 67;
        this->dirMode       = dirMode;
        this->flatBatchFill = flatBatchFill;
    }
};

using GPU_LSTM_dropout_FP16 = GPU_LSTM_dropout_test<half_float::half>;

TEST_P(GPU_LSTM_dropout_FP16, HalfTest) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(Full, GPU_LSTM_dropout_FP16, testing::ValuesIn(GetTestCases()));

using GPU_LSTM_dropout_FP32 = GPU_LSTM_dropout_test<float>;

TEST_P(GPU_LSTM_dropout_FP32, FloatTest) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(Full, GPU_LSTM_dropout_FP32, testing::ValuesIn(GetTestCases()));
