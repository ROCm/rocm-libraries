// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <half/half.hpp>
#include <vector>
#include "pooling2d_common.hpp"

namespace {

std::vector<pooling2d_gtest::PoolingTestCase> GetPooling2dWideTestCases()
{
    static std::vector<pooling2d_gtest::PoolingTestCase> cached_test_cases;
    static bool cached = false;

    if(cached)
    {
        return cached_test_cases;
    }

    std::vector<pooling2d_gtest::PoolingTestCase> test_cases;

    // Dataset 2: Wide window configurations
    // Match ctest's generate_multi_data_limited(..., 9)
    // For Dataset 2, there are 4 shapes, so no actual limiting happens (4 < 9),
    // but we use the same generation pattern for consistency.
    std::vector<std::vector<int>> dataset2_inputs = {
        {1, 3, 255, 255}, {2, 3, 227, 227}, {1, 7, 127, 127}, {1, 1, 410, 400}};

    // Lens: {{35, 35}, {100, 100}, {255, 255}, {410, 400}} - wide window kernel sizes
    std::vector<std::vector<int>> dataset2_lens = {{35, 35}, {100, 100}, {255, 255}, {410, 400}};

    // Strides: {{1, 1}} - only stride 1 for wide windows
    std::vector<std::vector<int>> dataset2_strides = {{1, 1}};

    // Pads: {{0, 0}} - no padding for wide windows
    std::vector<std::vector<int>> dataset2_pads = {{0, 0}};

    // Match ctest: Dataset 2 only uses miopenIndexUint32
    std::vector<miopenIndexType_t> dataset2_index_types = {miopenIndexUint32};
    std::vector<miopenPoolingMode_t> modes              = {
                     miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};
    std::vector<int> wsidx_values = {0, 1};

    for(const auto& in_shape : dataset2_inputs)
    {
        pooling2d_gtest::AddTestCasesForInput(in_shape,
                                              dataset2_lens,
                                              dataset2_strides,
                                              dataset2_pads,
                                              dataset2_index_types,
                                              modes,
                                              wsidx_values,
                                              test_cases,
                                              false, // skip_wide_check
                                              true); // is_wide_dataset
    }

    // Cache the results
    cached_test_cases = test_cases;
    cached            = true;

    return test_cases;
}

} // anonymous namespace

// Derived classes for Dataset 2 (wide window pooling)
class GPU_WidePooling2d_FP32 : public pooling2d_gtest::Pooling2dCommon<float>
{
};

class GPU_WidePooling2d_FP16 : public pooling2d_gtest::Pooling2dCommon<half_float::half>
{
};

TEST_P(GPU_WidePooling2d_FP32, FloatTest_pooling2d_wide) { this->RunTest(); }

TEST_P(GPU_WidePooling2d_FP16, HalfTest_pooling2d_wide) { this->RunTest(); }

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_WidePooling2d_FP32,
                         testing::ValuesIn(GetPooling2dWideTestCases()),
                         pooling2d_gtest::GetPoolingTestCaseName);

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_WidePooling2d_FP16,
                         testing::ValuesIn(GetPooling2dWideTestCases()),
                         pooling2d_gtest::GetPoolingTestCaseName);
