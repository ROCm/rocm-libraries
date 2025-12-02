/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include "gtest_common.hpp"

struct TernaryTensorOpsTestCase
{
    std::vector<size_t> tensorlens_ac;
    std::vector<size_t> tensorlens_b;
    std::vector<int64_t> offsets;
    std::vector<size_t> stride_a;
    std::vector<size_t> stride_b;
    std::vector<size_t> stride_c;
    std::vector<float> alphabeta;
    bool packed;
    miopenTensorOp_t operation;

    // friend std::ostream& operator<<(std::ostream& os, const TernaryTensorOpsTestCase& tc) {
    //     return os << "AC lens:" << tensor
    // }
};

template <typename T,
          miopenUnitUnderTest_t UUT    = miopenUnitNaiveGPU,
          miopenTestReference_t REF    = miopenTestReferenceOptimizedCPU,
          miopenAfterTestFailure_t ATF = miopenAfterTestFailureMoveOn>
struct TensorOpsCommon : public GTESTBase<UUT, REF, ATF>,
                         public testing::TestWithParam<TernaryTensorOpsTestCase>
{
protected:
    static void SetUpTestSuite()
    {
        if constexpr(!checkTestConfiguration(UUT, REF))
        {
            GTEST_SKIP() << "Test configuration is incorrect";
        }
    }

    void SetUp() override { prng::reset_seed(); }

    miopenStatus_t runOptimizedGPU() override
    {
        std::cout << "runOptimizedGPU()\n";
        return miopenStatusNotImplemented;
    }
    miopenStatus_t runNaiveGPU() override
    {
        std::cout << "runNaiveGPU()\n";
        return miopenStatusSuccess;
    }
    miopenStatus_t runOptimizedCPU() override
    {
        std::cout << "runOptimizedCPU()\n";
        return miopenStatusSuccess;
    }
    miopenStatus_t runNaiveCPU() override
    {
        std::cout << "runNaiveCPU()\n";
        return miopenStatusSuccess;
    }

    std::pair<bool, std::unordered_map<std::string, double>> verifyOptimizedGPU() override
    {
        return {true, std::unordered_map<std::string, double>()};
    };
    std::pair<bool, std::unordered_map<std::string, double>> verifyNaiveGPU() override
    {
        return {true, std::unordered_map<std::string, double>()};
    };
    std::pair<bool, std::unordered_map<std::string, double>> verifyOptimizedCPU() override
    {
        return {true, std::unordered_map<std::string, double>()};
    };

public:
    void run() { this->runTest(); }
};

using GPU_TernaryTensorOpsNew_FP32 = TensorOpsCommon<float>;

TEST_F(GPU_TernaryTensorOpsNew_FP32, TestFloat) { this->run(); }
