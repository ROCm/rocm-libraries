// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "pooling_common.hpp"

#define WORKAROUND_ISSUE_1670 1

namespace {

using PoolingTestCase = pooling_gtest::PoolingTestCase;

std::vector<PoolingTestCase> GetAsymPooling2dTestCases()
{
    static std::vector<PoolingTestCase> cached_test_cases;
    static bool cached = false;

    if(cached)
    {
        return cached_test_cases;
    }

    std::vector<PoolingTestCase> test_cases;

    // Dataset 1: Asymmetric dataset
    // Legacy test_drive coverage:
    //   forward path:  test_pooling2d --all --dataset 1 --limit 0
    //   backward path: test_pooling2d --forw 0
    std::vector<std::vector<int>> dataset1_inputs  = {{1, 4, 4, 4}};
    std::vector<std::vector<int>> dataset1_lens    = {{2, 2}, {1, 2}, {2, 1}};
    std::vector<std::vector<int>> dataset1_strides = {{1, 1}, {2, 1}, {1, 2}, {2, 2}};
#if WORKAROUND_ISSUE_1670
    std::vector<std::vector<int>> dataset1_pads = {{0, 0}};
#else
    std::vector<std::vector<int>> dataset1_pads = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
#endif

    std::vector<miopenIndexType_t> index_types = {
        miopenIndexUint8, miopenIndexUint16, miopenIndexUint32, miopenIndexUint64};
    std::vector<miopenPoolingMode_t> modes = {
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
                                            index_types,
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

class GPU_AsymPooling2d_FP32 : public pooling_gtest::PoolingCommon<float>
{
};

class GPU_AsymPooling2d_FP16 : public pooling_gtest::PoolingCommon<half_float::half>
{
};

class GPU_AsymPooling2d_BFP16 : public pooling_gtest::PoolingCommon<bfloat16>
{
};

TEST_P(GPU_AsymPooling2d_FP32, FloatTest) { RunTest(); }

TEST_P(GPU_AsymPooling2d_FP16, HalfTest) { RunTest(); }

TEST_P(GPU_AsymPooling2d_BFP16, BFloat16Test) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_AsymPooling2d_FP32,
                         testing::ValuesIn(GetAsymPooling2dTestCases()),
                         pooling_gtest::GetPoolingTestCaseName);

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_AsymPooling2d_FP16,
                         testing::ValuesIn(GetAsymPooling2dTestCases()),
                         pooling_gtest::GetPoolingTestCaseName);

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_AsymPooling2d_BFP16,
                         testing::ValuesIn(GetAsymPooling2dTestCases()),
                         pooling_gtest::GetPoolingTestCaseName);
