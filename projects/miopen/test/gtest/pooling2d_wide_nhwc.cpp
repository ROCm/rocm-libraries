/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include "pooling_common.hpp"

namespace pooling2d_wide_nhwc {

using namespace pooling_gtest;

struct GPU_WidePooling2d_NHWC_FP32 : public PoolingCommon<float> {};
struct GPU_WidePooling2d_NHWC_FP16 : public PoolingCommon<half_float::half> {};
struct GPU_WidePooling2d_NHWC_BFP16 : public PoolingCommon<bfloat16> {};

std::vector<PoolingTestCase> GetTestCases(miopenDataType_t prec)
{
    std::vector<int> wsidx_values = {0, 1};
    std::vector<miopenPoolingMode_t> modes = {miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};
    std::vector<miopenIndexType_t> index_types = {miopenIndexUint8, miopenIndexUint16, miopenIndexUint32, miopenIndexUint64};

    // Dataset 2: Wide window configs
    std::vector<std::vector<int>> in_shapes = {{1, 3, 255, 255}, {2, 3, 227, 227}, {1, 7, 127, 127}, {1, 1, 410, 400}};
    std::vector<std::vector<int>> lens_list = {{35, 35}, {100, 100}, {255, 255}, {410, 400}};
    std::vector<std::vector<int>> strides_list = {{1, 1}};
    std::vector<std::vector<int>> pads_list = {{0, 0}};

    std::vector<PoolingTestCase> test_cases;
    int num_uint16_case = 0, num_uint32_case = 0, num_uint32_case_imgidx = 0;
    int num_uint64_case = 0, num_uint64_case_imgidx = 0;

    for(const auto& in_shape : in_shapes)
    {
        AddTestCasesForInput(in_shape, lens_list, strides_list, pads_list, index_types, modes, wsidx_values, test_cases,
                             num_uint16_case, num_uint32_case, num_uint32_case_imgidx, num_uint64_case, num_uint64_case_imgidx,
                             false, true, "NHWC");
    }
    return test_cases;
}

} // namespace pooling2d_wide_nhwc

using namespace pooling2d_wide_nhwc;

TEST_P(GPU_WidePooling2d_NHWC_FP32, FloatTest) { RunTest(); }
TEST_P(GPU_WidePooling2d_NHWC_FP16, HalfTest) { RunTest(); }
TEST_P(GPU_WidePooling2d_NHWC_BFP16, BFloat16Test) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_WidePooling2d_NHWC_FP32,
                         testing::ValuesIn(GetTestCases(miopenFloat)),
                         GetPoolingTestCaseName);

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_WidePooling2d_NHWC_FP16,
                         testing::ValuesIn(GetTestCases(miopenHalf)),
                         GetPoolingTestCaseName);

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_WidePooling2d_NHWC_BFP16,
                         testing::ValuesIn(GetTestCases(miopenBFloat16)),
                         GetPoolingTestCaseName);
