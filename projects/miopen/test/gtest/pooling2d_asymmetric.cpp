// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <gtest/gtest.h>
#include <half/half.hpp>
#include "pooling2d_common.hpp"

namespace {

std::vector<pooling2d_gtest::PoolingTestCase> GetPooling2dAsymmetricTestCases()
{
    static std::vector<pooling2d_gtest::PoolingTestCase> cached_test_cases;
    static bool cached = false;

    if(cached)
    {
        return cached_test_cases;
    }

    std::vector<pooling2d_gtest::PoolingTestCase> test_cases;

    // Dataset 1: Asymmetric configurations
    // Match ctest's generate_multi_data_limited(..., 9)
    // For Dataset 1, there is only 1 shape, so no actual limiting happens,
    // but we use the same generation pattern for consistency.
    std::vector<std::vector<int>> dataset1_inputs = {{1, 4, 4, 4}};

    // Lens: {{2, 2}, {1, 2}, {2, 1}} - asymmetric kernel sizes
    std::vector<std::vector<int>> dataset1_lens = {{2, 2}, {1, 2}, {2, 1}};

    // Strides: {{1, 1}, {2, 1}, {1, 2}, {2, 2}} - asymmetric strides
    std::vector<std::vector<int>> dataset1_strides = {{1, 1}, {2, 1}, {1, 2}, {2, 2}};

    // Pads: controlled by WORKAROUND_ISSUE_1670 (matching original ctest behavior)
#if WORKAROUND_ISSUE_1670
    std::vector<std::vector<int>> dataset1_pads = {{0, 0}};
#else
    std::vector<std::vector<int>> dataset1_pads = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
#endif

    // Match ctest: Dataset 1 only uses miopenIndexUint32
    std::vector<miopenIndexType_t> dataset1_index_types = {miopenIndexUint32};
    std::vector<miopenPoolingMode_t> modes              = {
                     miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};
    std::vector<int> wsidx_values = {0, 1};

    for(const auto& in_shape : dataset1_inputs)
    {
        pooling2d_gtest::AddTestCasesForInput(in_shape,
                                              dataset1_lens,
                                              dataset1_strides,
                                              dataset1_pads,
                                              dataset1_index_types,
                                              modes,
                                              wsidx_values,
                                              test_cases,
                                              true,   // skip_wide_check
                                              false); // is_wide_dataset
    }

    // Cache the results
    cached_test_cases = test_cases;
    cached            = true;

    return test_cases;
}

} // anonymous namespace

// Derived classes for Dataset 1 (asymmetric pooling)
class GPU_AsymPooling2d_FP32 : public pooling2d_gtest::Pooling2dCommon<float>
{
};

class GPU_AsymPooling2d_FP16 : public pooling2d_gtest::Pooling2dCommon<half_float::half>
{
};

TEST_P(GPU_AsymPooling2d_FP32, FloatTest_pooling2d_asymmetric) { this->RunTest(); }

TEST_P(GPU_AsymPooling2d_FP16, HalfTest_pooling2d_asymmetric) { this->RunTest(); }

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_AsymPooling2d_FP32,
                         testing::ValuesIn(GetPooling2dAsymmetricTestCases()),
                         pooling2d_gtest::GetPoolingTestCaseName);

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_AsymPooling2d_FP16,
                         testing::ValuesIn(GetPooling2dAsymmetricTestCases()),
                         pooling2d_gtest::GetPoolingTestCaseName);
