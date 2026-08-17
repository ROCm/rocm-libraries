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

namespace {

struct TestCase
{
    TestCase(int batch_size_, int seq_len_, int vector_len_, int hidden_size_)
        : batch_size(batch_size_),
          seq_len(seq_len_),
          vector_len(vector_len_),
          hidden_size(hidden_size_)
    {
    }

    int batch_size;
    int seq_len;
    int vector_len;
    int hidden_size;

    friend std::ostream& operator<<(std::ostream& os, const TestCase& tc)
    {
        return os << "batch-size:" << tc.batch_size << " seq-len:" << tc.seq_len
                  << " vector-len:" << tc.vector_len << " hidden-size:" << tc.hidden_size;
    }
};

std::vector<TestCase> GetTestCases()
{
    return std::vector{
        // clang-format off
        //     batch-size seq-len vector-len hidden-size
        TestCase( 16,       25,      512,       512 ),
        TestCase( 32,       25,      512,       512 ),
        TestCase( 64,       25,      512,       512 ),
        TestCase(128,       25,      512,       512 ),
        TestCase( 16,       25,     1024,      1024 ),
        TestCase( 32,       25,     1024,      1024 ),
        TestCase( 64,       25,     1024,      1024 ),
        TestCase(128,       25,     1024,      1024 ),
        TestCase( 16,       25,     2048,      2048 ),
        TestCase( 32,       25,     2048,      2048 ),
        TestCase( 64,       25,     2048,      2048 ),
        TestCase(128,       25,     2048,      2048 ),
        TestCase( 16,       25,     4096,      4096 ),
        TestCase( 32,       25,     4096,      4096 ),
        TestCase( 64,       25,     4096,      4096 ),
        TestCase(128,       25,     4096,      4096 ),
        TestCase(  8,       50,     1536,      1536 ),
        TestCase( 16,       50,     1536,      1536 ),
        TestCase( 32,       50,     1536,      1536 ),
        TestCase( 16,      150,      256,       256 ),
        TestCase( 32,      150,      256,       256 ),
        TestCase( 64,      150,      256,       256 )
        // clang-format on
    };
}

} // namespace

template <typename T>
struct GPU_DeepBench_LSTM_test : LSTM_test<T>, testing::TestWithParam<TestCase>
{
protected:
    void SetUp() override
    {
        this->dataType = miopen_type<T>{};

        auto [batchSize, seqLength, inVecLen, hiddenSize] = GetParam();

        this->numLayers     = 1;
        this->inputMode     = 1;
        this->biasMode      = 0;
        this->dirMode       = 0;
        this->flatBatchFill = 1;
        this->batchSize     = batchSize;
        this->seqLength     = seqLength;
        this->inVecLen      = inVecLen;
        this->hiddenSize    = hiddenSize;
    }
};

using GPU_DeepBench_LSTM_FP16 = GPU_DeepBench_LSTM_test<half_float::half>;

TEST_P(GPU_DeepBench_LSTM_FP16, HalfTest) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(Full, GPU_DeepBench_LSTM_FP16, testing::ValuesIn(GetTestCases()));

using GPU_DeepBench_LSTM_FP32 = GPU_DeepBench_LSTM_test<float>;

TEST_P(GPU_DeepBench_LSTM_FP32, FloatTest) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(Full, GPU_DeepBench_LSTM_FP32, testing::ValuesIn(GetTestCases()));
