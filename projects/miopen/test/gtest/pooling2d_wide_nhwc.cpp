// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>
#include <gtest/gtest.h>
#include <half/half.hpp>
#include "pooling2d_common.hpp"

namespace {

std::vector<pooling2d_gtest::PoolingTestCase> GetPooling2dWideNHWCTestCases()
{
    static std::vector<pooling2d_gtest::PoolingTestCase> cached_test_cases;
    static bool cached = false;

    if(cached)
    {
        return cached_test_cases;
    }

    std::vector<pooling2d_gtest::PoolingTestCase> test_cases;

    std::vector<std::vector<int>> dataset2_inputs = {
        {1, 3, 255, 255}, {2, 3, 227, 227}, {1, 7, 127, 127}, {1, 1, 410, 400}};

    std::vector<std::vector<int>> dataset2_lens = {{35, 35}, {100, 100}, {255, 255}, {410, 400}};
    std::vector<std::vector<int>> dataset2_strides = {{1, 1}};
    std::vector<std::vector<int>> dataset2_pads = {{0, 0}};

    std::vector<miopenIndexType_t> dataset2_index_types = {miopenIndexUint32};
    std::vector<miopenPoolingMode_t> modes = {
        miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};
    std::vector<int> wsidx_values = {0, 1};

    int num_uint16_case = 0, num_uint32_case = 0, num_uint32_case_imgidx = 0;
    int num_uint64_case = 0, num_uint64_case_imgidx = 0;

    for(const auto& input_dims : dataset2_inputs)
    {
        pooling2d_gtest::AddTestCasesForInput(input_dims,
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
                                              false,
                                              true,
                                              "NHWC",
                                              "NHWC");
    }

    cached_test_cases = test_cases;
    cached            = true;
    return test_cases;
}

} // anonymous namespace

class GPU_WidePooling2d_NHWC_FP32 : public pooling2d_gtest::Pooling2dCommon<float>
{
};
class GPU_WidePooling2d_NHWC_FP16 : public pooling2d_gtest::Pooling2dCommon<half_float::half>
{
};
class GPU_WidePooling2d_NHWC_BFP16 : public pooling2d_gtest::Pooling2dCommon<bfloat16>
{
};

TEST_P(GPU_WidePooling2d_NHWC_FP32, FloatTest_pooling2d_wide_nhwc) { this->RunTest(); }
TEST_P(GPU_WidePooling2d_NHWC_FP16, HalfTest_pooling2d_wide_nhwc) { this->RunTest(); }
TEST_P(GPU_WidePooling2d_NHWC_BFP16, BFloat16Test_pooling2d_wide_nhwc) { this->RunTest(); }

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_WidePooling2d_NHWC_FP32,
                         testing::ValuesIn(GetPooling2dWideNHWCTestCases()),
                         pooling2d_gtest::GetPoolingTestCaseName);
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_WidePooling2d_NHWC_FP16,
                         testing::ValuesIn(GetPooling2dWideNHWCTestCases()),
                         pooling2d_gtest::GetPoolingTestCaseName);
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_WidePooling2d_NHWC_BFP16,
                         testing::ValuesIn(GetPooling2dWideNHWCTestCases()),
                         pooling2d_gtest::GetPoolingTestCaseName);
