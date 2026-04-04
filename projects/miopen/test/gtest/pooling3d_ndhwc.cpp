// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "pooling_common.hpp"

namespace {

using PoolingTestCase = pooling_gtest::PoolingTestCase;

std::vector<PoolingTestCase> GetNDHWCPooling3dTestCases()
{
    static std::vector<PoolingTestCase> cached_test_cases;
    static bool cached = false;

    if(cached)
    {
        return cached_test_cases;
    }

    std::vector<PoolingTestCase> test_cases;

    // Keep parity with legacy ctest NDHWC coverage:
    // input=[16,64,3,4,4], lens=[2,2,2], strides=[2,2,2], pads=[1,1,1],
    // mode=max, index_type=uint32, wsidx=1.
    std::vector<std::vector<int>> dataset0_inputs  = {{16, 64, 3, 4, 4}};
    std::vector<std::vector<int>> dataset0_lens    = {{2, 2, 2}};
    std::vector<std::vector<int>> dataset0_strides = {{2, 2, 2}};
    std::vector<std::vector<int>> dataset0_pads    = {{1, 1, 1}};

    std::vector<miopenIndexType_t> index_types = {miopenIndexUint32};
    std::vector<miopenPoolingMode_t> modes     = {miopenPoolingMax};

    std::vector<int> wsidx_values = {1};

    int num_uint16_case = 0, num_uint32_case = 0, num_uint32_case_imgidx = 0;
    int num_uint64_case = 0, num_uint64_case_imgidx = 0;

    for(const auto& in_shape : dataset0_inputs)
    {
        pooling_gtest::AddTestCasesForInput(in_shape,
                                              dataset0_lens,
                                              dataset0_strides,
                                              dataset0_pads,
                                              index_types,
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
                                              "NDHWC");
    }

    cached_test_cases = test_cases;
    cached            = true;
    return test_cases;
}

} // namespace

class GPU_Pooling3d_NDHWC_FP32 : public pooling_gtest::PoolingCommon<float, 3>
{
};

class GPU_Pooling3d_NDHWC_FP16 : public pooling_gtest::PoolingCommon<half_float::half, 3>
{
};

class GPU_Pooling3d_NDHWC_BFP16 : public pooling_gtest::PoolingCommon<bfloat16, 3>
{
};

TEST_P(GPU_Pooling3d_NDHWC_FP32, FloatTest) { RunTest(); }

TEST_P(GPU_Pooling3d_NDHWC_FP16, HalfTest) { RunTest(); }

TEST_P(GPU_Pooling3d_NDHWC_BFP16, BFloat16Test) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Pooling3d_NDHWC_FP32,
                         testing::ValuesIn(GetNDHWCPooling3dTestCases()),
                         pooling_gtest::GetPoolingTestCaseName);

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Pooling3d_NDHWC_FP16,
                         testing::ValuesIn(GetNDHWCPooling3dTestCases()),
                         pooling_gtest::GetPoolingTestCaseName);

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Pooling3d_NDHWC_BFP16,
                         testing::ValuesIn(GetNDHWCPooling3dTestCases()),
                         pooling_gtest::GetPoolingTestCaseName);
