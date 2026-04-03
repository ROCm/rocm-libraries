// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "pooling_common.hpp"

namespace {

using PoolingTestCase = pooling_gtest::PoolingTestCase;

std::vector<PoolingTestCase> GetPooling2dWideTestCases()
{
    static std::vector<PoolingTestCase> cached_test_cases;
    static bool cached = false;

    if(cached)
    {
        return cached_test_cases;
    }

    std::vector<PoolingTestCase> test_cases;

    // Dataset 2: Wide window configurations
    std::vector<std::vector<int>> dataset2_inputs = {
        {1, 3, 255, 255}, {2, 3, 227, 227}, {1, 7, 127, 127}, {1, 1, 410, 400}};

    std::vector<std::vector<int>> dataset2_lens = {{35, 35}, {100, 100}, {255, 255}, {410, 400}};
    std::vector<std::vector<int>> dataset2_strides = {{1, 1}};
    std::vector<std::vector<int>> dataset2_pads = {{0, 0}};

    std::vector<miopenIndexType_t> dataset2_index_types = {miopenIndexUint32};
    std::vector<miopenPoolingMode_t> modes              = {
                     miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};
    std::vector<int> wsidx_values = {0, 1};

    int num_uint16_case = 0, num_uint32_case = 0, num_uint32_case_imgidx = 0;
    int num_uint64_case = 0, num_uint64_case_imgidx = 0;

    for(const auto& in_shape : dataset2_inputs)
    {
        pooling_gtest::AddTestCasesForInput(in_shape,
                                              dataset2_lens,
                                              dataset2_strides,
                                              dataset2_pads,
                                              dataset2_index_types,
                                              modes,
                                              wsidx_values,
                                              test_cases,
                                              num_uint16_case,
                                              num_uint32_case,
                                              num_uint32_case_imgidx,
                                              num_uint64_case,
                                              num_uint64_case_imgidx,
                                              false,
                                              true,
                                              "NCHW");
    }

    cached_test_cases = test_cases;
    cached            = true;
    return test_cases;
}

} // anonymous namespace

// Derived classes for Dataset 2 (wide window pooling)
class GPU_WidePooling2d_FP32 : public pooling_gtest::PoolingCommon<float>
{
};

class GPU_WidePooling2d_FP16 : public pooling_gtest::PoolingCommon<half_float::half>
{
};

TEST_P(GPU_WidePooling2d_FP32, FloatTest_pooling2d_wide) { this->RunTest(); }

TEST_P(GPU_WidePooling2d_FP16, HalfTest_pooling2d_wide) { this->RunTest(); }

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_WidePooling2d_FP32,
                         testing::ValuesIn(GetPooling2dWideTestCases()),
                         pooling_gtest::GetPoolingTestCaseName);

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_WidePooling2d_FP16,
                         testing::ValuesIn(GetPooling2dWideTestCases()),
                         pooling_gtest::GetPoolingTestCaseName);
