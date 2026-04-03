// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "pooling_common.hpp"

namespace {

using PoolingTestCase = pooling_gtest::PoolingTestCase;

std::vector<PoolingTestCase> GetPooling2dAsymmetricTestCases()
{
    static std::vector<PoolingTestCase> cached_test_cases;
    static bool cached = false;

    if(cached)
    {
        return cached_test_cases;
    }

    std::vector<PoolingTestCase> test_cases;

    // Dataset 1: Asymmetric configurations
    std::vector<std::vector<int>> dataset1_inputs = {{1, 4, 4, 4}};
    std::vector<std::vector<int>> dataset1_lens = {{2, 2}, {1, 2}, {2, 1}};
    std::vector<std::vector<int>> dataset1_strides = {{1, 1}, {2, 1}, {1, 2}, {2, 2}};

#if WORKAROUND_ISSUE_1670
    std::vector<std::vector<int>> dataset1_pads = {{0, 0}};
#else
    std::vector<std::vector<int>> dataset1_pads = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
#endif

    std::vector<miopenIndexType_t> dataset1_index_types = {miopenIndexUint32};
    std::vector<miopenPoolingMode_t> modes              = {
                     miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};
    std::vector<int> wsidx_values = {0, 1};

    int num_uint16_case = 0, num_uint32_case = 0, num_uint32_case_imgidx = 0;
    int num_uint64_case = 0, num_uint64_case_imgidx = 0;

    for(const auto& in_shape : dataset1_inputs)
    {
        pooling_gtest::AddTestCasesForInput(in_shape,
                                              dataset1_lens,
                                              dataset1_strides,
                                              dataset1_pads,
                                              dataset1_index_types,
                                              modes,
                                              wsidx_values,
                                              test_cases,
                                              num_uint16_case,
                                              num_uint32_case,
                                              num_uint32_case_imgidx,
                                              num_uint64_case,
                                              num_uint64_case_imgidx,
                                              false,
                                              false,
                                              "NCHW");
    }

    cached_test_cases = test_cases;
    cached            = true;
    return test_cases;
}

} // anonymous namespace

// Derived classes for Dataset 1 (asymmetric pooling)
class GPU_AsymPooling2d_FP32 : public pooling_gtest::PoolingCommon<float>
{
};

class GPU_AsymPooling2d_FP16 : public pooling_gtest::PoolingCommon<half_float::half>
{
};

TEST_P(GPU_AsymPooling2d_FP32, FloatTest_pooling2d_asymmetric) { this->RunTest(); }

TEST_P(GPU_AsymPooling2d_FP16, HalfTest_pooling2d_asymmetric) { this->RunTest(); }

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_AsymPooling2d_FP32,
                         testing::ValuesIn(GetPooling2dAsymmetricTestCases()),
                         pooling_gtest::GetPoolingTestCaseName);

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_AsymPooling2d_FP16,
                         testing::ValuesIn(GetPooling2dAsymmetricTestCases()),
                         pooling_gtest::GetPoolingTestCaseName);
