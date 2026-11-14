// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "lstm.hpp"

// The gradient of the bias of an RNN is the sum over all rows of the workspace (batch size times
// sequence length of them) and is computed with miopenReduceTensor. It takes the multi-block path
// of the reduction (the partial results are stored in the workspace of the reduction) when there
// are more than 1024 rows. The other LSTM tests have far fewer rows, and most of them are in the
// Full tier. The data of the LSTM test is small, so this test checks that the path runs and that
// its result is correct (the harness of lstm.hpp prints a mismatch, and the test fails if it
// printed one); reduce_multiblock.cpp checks that its partial results are not clipped.
namespace {

struct TestCase
{
    TestCase(int dirMode_, int algoMode_) : dirMode(dirMode_), algoMode(algoMode_) {}

    int dirMode;
    int algoMode;

    friend std::ostream& operator<<(std::ostream& os, const TestCase& tc)
    {
        return os << "dir-mode:" << tc.dirMode << " algo-mode:" << tc.algoMode;
    }
};

std::vector<TestCase> GetTestCases()
{
    return std::vector{
        // clang-format off
        //       dir-mode algo-mode
        TestCase(0,       0),
        TestCase(0,       1),
        TestCase(1,       0),
        TestCase(1,       1),
        // clang-format on
    };
}

} // namespace

template <typename T>
struct GPU_LSTMBiasMultiBlock_test : LSTM_test<T>, testing::TestWithParam<TestCase>
{
protected:
    void SetUp() override
    {
        this->dataType = miopen_type<T>{};

        auto [dirMode, algoMode] = GetParam();

        // 40 * 30 = 1200 rows
        this->batchSize  = 40;
        this->seqLength  = 30;
        this->batchSeq   = std::vector<int>(this->seqLength, this->batchSize);
        this->inVecLen   = 8;
        this->hiddenSize = 8;
        this->numLayers  = 1;
        this->inputMode  = 0;
        this->biasMode   = 1;
        this->dirMode    = dirMode;
        this->algoMode   = algoMode;
    }

    void RunCheckedTest()
    {
        this->RunTest();
        EXPECT_EQ(this->verifyFailures, 0u)
            << "the CPU and the GPU results of the LSTM differ, or the GPU run threw";
    }
};

using GPU_LSTMBiasMultiBlock_FP16 = GPU_LSTMBiasMultiBlock_test<half_float::half>;

TEST_P(GPU_LSTMBiasMultiBlock_FP16, HalfTest) { RunCheckedTest(); }

INSTANTIATE_TEST_SUITE_P(Standard, GPU_LSTMBiasMultiBlock_FP16, testing::ValuesIn(GetTestCases()));

using GPU_LSTMBiasMultiBlock_FP32 = GPU_LSTMBiasMultiBlock_test<float>;

TEST_P(GPU_LSTMBiasMultiBlock_FP32, FloatTest) { RunCheckedTest(); }

// the FP32 reduction has the same compute and input type; one case is a control
INSTANTIATE_TEST_SUITE_P(Standard, GPU_LSTMBiasMultiBlock_FP32, testing::Values(TestCase(0, 0)));
