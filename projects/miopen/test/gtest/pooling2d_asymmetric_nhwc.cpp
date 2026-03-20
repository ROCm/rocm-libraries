// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>
#include <gtest/gtest.h>
#include <half/half.hpp>
#include "pooling2d_common.hpp"

namespace {

std::vector<pooling2d_gtest::PoolingTestCase> GetPooling2dAsymmetricNHWCTestCases()
{
    static std::vector<pooling2d_gtest::PoolingTestCase> cached_test_cases;
    static bool cached = false;

    if(cached)
    {
        return cached_test_cases;
    }

    std::vector<pooling2d_gtest::PoolingTestCase> test_cases;

    std::vector<std::vector<int>> dataset1_inputs = {{1, 4, 4, 4}};
    std::vector<std::vector<int>> dataset1_lens    = {{2, 2}, {1, 2}, {2, 1}};
    std::vector<std::vector<int>> dataset1_strides = {{1, 1}, {2, 1}, {1, 2}, {2, 2}};
#if WORKAROUND_ISSUE_1670
    std::vector<std::vector<int>> dataset1_pads = {{0, 0}};
#else
    std::vector<std::vector<int>> dataset1_pads = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
#endif

    std::vector<miopenIndexType_t> dataset1_index_types = {miopenIndexUint32};
    std::vector<miopenPoolingMode_t> modes = {
        miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};
    std::vector<int> wsidx_values = {0, 1};

    for(const auto& input_dims : dataset1_inputs)
    {
        pooling2d_gtest::AddTestCasesForInput(input_dims,
                                              dataset1_lens,
                                              dataset1_strides,
                                              dataset1_pads,
                                              dataset1_index_types,
                                              modes,
                                              wsidx_values,
                                              test_cases,
                                              true,   // skip_wide_check
                                              false,  // is_wide_dataset
                                              "NHWC",
                                              "NHWC");
    }

    cached_test_cases = test_cases;
    cached            = true;
    return test_cases;
}

} // anonymous namespace

class GPU_AsymPooling2d_NHWC_FP32 : public pooling2d_gtest::Pooling2dCommon<float>
{
};
class GPU_AsymPooling2d_NHWC_FP16 : public pooling2d_gtest::Pooling2dCommon<half_float::half>
{
};
class GPU_AsymPooling2d_NHWC_BFP16 : public pooling2d_gtest::Pooling2dCommon<bfloat16>
{
};

TEST_P(GPU_AsymPooling2d_NHWC_FP32, FloatTest_pooling2d_asymmetric_nhwc) { this->RunTest(); }
TEST_P(GPU_AsymPooling2d_NHWC_FP16, HalfTest_pooling2d_asymmetric_nhwc) { this->RunTest(); }
TEST_P(GPU_AsymPooling2d_NHWC_BFP16, BFloat16Test_pooling2d_asymmetric_nhwc) { this->RunTest(); }

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_AsymPooling2d_NHWC_FP32,
                         testing::ValuesIn(GetPooling2dAsymmetricNHWCTestCases()),
                         pooling2d_gtest::GetPoolingTestCaseName);
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_AsymPooling2d_NHWC_FP16,
                         testing::ValuesIn(GetPooling2dAsymmetricNHWCTestCases()),
                         pooling2d_gtest::GetPoolingTestCaseName);
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_AsymPooling2d_NHWC_BFP16,
                         testing::ValuesIn(GetPooling2dAsymmetricNHWCTestCases()),
                         pooling2d_gtest::GetPoolingTestCaseName);
