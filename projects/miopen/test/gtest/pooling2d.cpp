// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "pooling_common.hpp"

#define WORKAROUND_ISSUE_1670 1
#define TEST_GET_INPUT_TENSOR 0

namespace {

using PoolingTestCase = pooling_gtest::PoolingTestCase;

std::vector<PoolingTestCase> GetPooling2dTestCases()
{
    static std::vector<PoolingTestCase> cached_test_cases;
    static bool cached = false;

    if(cached)
    {
        return cached_test_cases;
    }

    std::vector<PoolingTestCase> test_cases;

    // Dataset 0: Default dataset (various tensor sizes)
    std::vector<std::vector<int>> dataset0_inputs;
#if TEST_GET_INPUT_TENSOR
    int batch_factor                      = 0;
    std::set<std::vector<int>> in_dim_set = get_inputs<int>(batch_factor);
    dataset0_inputs.assign(in_dim_set.begin(), in_dim_set.end());
#else
    dataset0_inputs = {
        {1, 19, 1024, 2048}, {10, 3, 32, 32},     {5, 32, 8, 8},     {2, 1024, 12, 12},
        {4, 3, 231, 231},    {8, 3, 227, 227},    {1, 384, 13, 13},  {1, 96, 27, 27},
        {2, 160, 7, 7},      {1, 192, 256, 512},  {2, 192, 28, 28},  {1, 832, 64, 128},
        {1, 256, 56, 56},    {4, 3, 224, 224},    {2, 64, 112, 112}, {2, 608, 4, 4},
        {1, 2048, 11, 11},   {1, 16, 4096, 4096}, {1, 3, 8, 8},      {2, 16, 14, 14},
        {1, 32, 7, 7},       {4, 64, 4, 4},       {1, 3, 32, 32},    {2, 64, 56, 56},
        {1, 128, 28, 28},    {1, 3, 224, 224},    {1, 64, 112, 112}};
#endif
    std::vector<std::vector<int>> dataset0_lens         = {{2, 2}, {3, 3}};
    std::vector<std::vector<int>> dataset0_strides      = {{2, 2}, {1, 1}};
    std::vector<std::vector<int>> dataset0_pads         = {{0, 0}, {1, 1}};
    std::vector<miopenIndexType_t> dataset0_index_types = {
        miopenIndexUint8, miopenIndexUint16, miopenIndexUint32, miopenIndexUint64};
    std::vector<miopenPoolingMode_t> modes = {
        miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};
    std::vector<int> wsidx_values = {0, 1};

    int num_uint16_case = 0, num_uint32_case = 0, num_uint32_case_imgidx = 0;
    int num_uint64_case = 0, num_uint64_case_imgidx = 0;

    for(const auto& in_shape : dataset0_inputs)
    {
        pooling_gtest::AddTestCasesForInput(in_shape,
                                            dataset0_lens,
                                            dataset0_strides,
                                            dataset0_pads,
                                            dataset0_index_types,
                                            modes,
                                            wsidx_values,
                                            test_cases,
                                            num_uint16_case,
                                            num_uint32_case,
                                            num_uint32_case_imgidx,
                                            num_uint64_case,
                                            num_uint64_case_imgidx,
                                            true,
                                            false,
                                            "NCHW");
    }

    cached_test_cases = test_cases;
    cached            = true;
    return test_cases;
}

} // anonymous namespace

// Derived classes for Dataset 0 (standard pooling)
class GPU_Pooling2d_FP32 : public pooling_gtest::PoolingCommon<float>
{
};

class GPU_Pooling2d_FP16 : public pooling_gtest::PoolingCommon<half_float::half>
{
};

class GPU_Pooling2d_BFP16 : public pooling_gtest::PoolingCommon<bfloat16>
{
};

TEST_P(GPU_Pooling2d_FP32, FloatTest_pooling2d) { RunTest(); }

TEST_P(GPU_Pooling2d_FP16, HalfTest_pooling2d) { RunTest(); }

TEST_P(GPU_Pooling2d_BFP16, BFloat16Test_pooling2d) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Pooling2d_FP32,
                         testing::ValuesIn(GetPooling2dTestCases()),
                         pooling_gtest::GetPoolingTestCaseName);

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Pooling2d_FP16,
                         testing::ValuesIn(GetPooling2dTestCases()),
                         pooling_gtest::GetPoolingTestCaseName);

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Pooling2d_BFP16,
                         testing::ValuesIn(GetPooling2dTestCases()),
                         pooling_gtest::GetPoolingTestCaseName);
