/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include <gtest/gtest.h>
#include <miopen/env.hpp>
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "pooling2d_common.hpp"

namespace pooling3d_ndhwc {

using namespace pooling2d_gtest;

struct GPU_Pooling3d_NDHWC_FP32 : public Pooling2dCommon<float> {};
struct GPU_Pooling3d_NDHWC_FP16 : public Pooling2dCommon<half_float::half> {};
struct GPU_Pooling3d_NDHWC_BFP16 : public Pooling2dCommon<bfloat16> {};

std::vector<PoolingTestCase> GetTestCases(miopenDataType_t prec)
{
    std::vector<int> wsidx_values = {1};
    std::vector<miopenPoolingMode_t> modes = {miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};
    std::vector<miopenIndexType_t> index_types = {miopenIndexUint8, miopenIndexUint16, miopenIndexUint32, miopenIndexUint64};

    // 3D pooling shapes
    std::vector<std::vector<int>> in_shapes = {
        {16, 64, 3, 4, 4}, {16, 32, 4, 9, 9}, {8, 512, 3, 14, 14}, {8, 512, 4, 28, 28},
        {16, 64, 56, 56, 56}, {4, 3, 4, 227, 227}, {4, 4, 4, 161, 700}, {1, 3, 4, 4, 4},
        {2, 8, 2, 8, 8}, {1, 16, 3, 5, 5}, {1, 32, 4, 14, 14}, {2, 64, 8, 8, 8},
        {1, 16, 4, 28, 28}, {1, 3, 8, 56, 56}, {2, 64, 4, 28, 28}, {1, 32, 16, 32, 32}
    };
    std::vector<std::vector<int>> lens_list = {{2, 2, 2}, {3, 3, 3}, {1, 2, 2}};
    std::vector<std::vector<int>> strides_list = {{2, 2, 2}, {1, 1, 1}, {1, 2, 2}};
    std::vector<std::vector<int>> pads_list = {{0, 0, 0}, {1, 1, 1}};

    std::vector<PoolingTestCase> test_cases;
    for(const auto& in_shape : in_shapes)
    {
        AddTestCasesForInput(in_shape,
                             lens_list,
                             strides_list,
                             pads_list,
                             index_types,
                             modes,
                             wsidx_values,
                             test_cases,
                             false, // skip_wide_check
                             false, // is_wide_dataset
                             "NDHWC",
                             "NDHWC");
    }
    return test_cases;
}

} // namespace pooling3d_ndhwc

using namespace pooling3d_ndhwc;

TEST_P(GPU_Pooling3d_NDHWC_FP32, FloatTest) { RunTest(); }
TEST_P(GPU_Pooling3d_NDHWC_FP16, HalfTest) { RunTest(); }
TEST_P(GPU_Pooling3d_NDHWC_BFP16, BFloat16Test) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Pooling3d_NDHWC_FP32,
                         testing::ValuesIn(GetTestCases(miopenFloat)),
                         GetPoolingTestCaseName);

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Pooling3d_NDHWC_FP16,
                         testing::ValuesIn(GetTestCases(miopenHalf)),
                         GetPoolingTestCaseName);

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Pooling3d_NDHWC_BFP16,
                         testing::ValuesIn(GetTestCases(miopenBFloat16)),
                         GetPoolingTestCaseName);
