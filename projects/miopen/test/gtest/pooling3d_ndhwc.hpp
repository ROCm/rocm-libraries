/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef GUARD_MIOPEN_TEST_GTEST_POOLING3D_NDHWC_HPP
#define GUARD_MIOPEN_TEST_GTEST_POOLING3D_NDHWC_HPP

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

    // Dataset 0: Default dataset (3D NDHWC)
    std::vector<std::vector<int>> dataset0_inputs = {
        {16, 64, 3, 4, 4},
        {16, 32, 4, 9, 9},
        {8, 512, 3, 14, 14},
        {8, 512, 4, 28, 28},
        {16, 64, 56, 56, 56},
        {4, 3, 4, 227, 227},
        {4, 4, 4, 161, 700},
        {1, 3, 4, 4, 4},
        {2, 8, 2, 8, 8},
        {1, 16, 3, 5, 5},
        {1, 32, 4, 14, 14},
        {2, 64, 8, 8, 8},
        {1, 16, 4, 28, 28},
        {1, 3, 8, 56, 56},
        {2, 64, 4, 28, 28},
        {1, 32, 16, 32, 32}};

    std::vector<std::vector<int>> dataset0_lens    = {{2, 2, 2}, {3, 3, 3}, {1, 2, 2}};
    std::vector<std::vector<int>> dataset0_strides = {{2, 2, 2}, {1, 1, 1}, {1, 2, 2}};
    std::vector<std::vector<int>> dataset0_pads    = {{0, 0, 0}, {1, 1, 1}};

    std::vector<miopenIndexType_t> index_types = {
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

#endif
