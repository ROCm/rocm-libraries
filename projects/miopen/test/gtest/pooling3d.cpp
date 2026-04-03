// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <half/half.hpp>

#include <miopen/logger.hpp>
#include <miopen/tensor_layout.hpp>

#include "pooling_common.hpp"

namespace {

using PoolingTestCase = pooling_gtest::PoolingTestCase;

std::vector<PoolingTestCase> GetPooling3dTestCases()
{
    static std::vector<PoolingTestCase> cached_test_cases;
    static bool cached = false;

    if(cached)
    {
        return cached_test_cases;
    }

    std::vector<PoolingTestCase> test_cases;

    // Dataset 0: Default dataset (various tensor sizes)
    std::vector<std::vector<int>> dataset0_inputs = {
        {16, 64, 3, 4, 4}, {16, 32, 4, 9, 9}, {8, 512, 3, 14, 14}, {8, 512, 4, 28, 28}};

    std::vector<std::vector<int>> dataset0_lens    = {{2, 2, 2}, {3, 3, 3}, {1, 2, 2}};
    std::vector<std::vector<int>> dataset0_strides = {{2, 2, 2}, {1, 1, 1}, {1, 2, 2}};
    std::vector<std::vector<int>> dataset0_pads    = {{0, 0, 0}, {1, 1, 1}};

    std::vector<miopenIndexType_t> dataset0_index_types = {
        miopenIndexUint8, miopenIndexUint16, miopenIndexUint32, miopenIndexUint64};
    std::vector<miopenPoolingMode_t> modes = {
        miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};

    std::vector<int> wsidx_values = {1};

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
                                              "NCDHW");
    }

    cached_test_cases = test_cases;
    cached            = true;
    return test_cases;
}

} // namespace

// Derived classes for Dataset 0 (standard 3D pooling)
class GPU_Pooling3d_FP32 : public pooling_gtest::PoolingCommon<float, 3>
{
};

class GPU_Pooling3d_FP16 : public pooling_gtest::PoolingCommon<half_float::half, 3>
{
};

class GPU_Pooling3d_BFP16 : public pooling_gtest::PoolingCommon<bfloat16, 3>
{
};

TEST_P(GPU_Pooling3d_FP32, Test) { RunTest(); }

TEST_P(GPU_Pooling3d_FP16, Test) { RunTest(); }

TEST_P(GPU_Pooling3d_BFP16, Test) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Pooling3d_FP32,
                         testing::ValuesIn(GetPooling3dTestCases()),
                         pooling_gtest::GetPoolingTestCaseName);

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Pooling3d_FP16,
                         testing::ValuesIn(GetPooling3dTestCases()),
                         pooling_gtest::GetPoolingTestCaseName);

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Pooling3d_BFP16,
                         testing::ValuesIn(GetPooling3dTestCases()),
                         pooling_gtest::GetPoolingTestCaseName);
