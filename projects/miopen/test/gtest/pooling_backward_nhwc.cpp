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

#include <gtest/gtest.h>
#include <miopen/env.hpp>
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "pooling2d.hpp"

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLAGS_ARGS)

namespace env = miopen::env;

namespace pooling_backward_nhwc {

// Test fixture for FP32 backward pooling with NHWC layout
class GPU_PoolingBackward_NHWC_FP32 : public testing::TestWithParam<std::vector<std::string>>
{
    MIOPEN_DECLARE_GTEST_USES_TEST_DRIVE();
};

// Test fixture for FP16 backward pooling with NHWC layout
class GPU_PoolingBackward_NHWC_FP16 : public testing::TestWithParam<std::vector<std::string>>
{
    MIOPEN_DECLARE_GTEST_USES_TEST_DRIVE();
};

// Test fixture for BF16 backward pooling with NHWC layout
class GPU_PoolingBackward_NHWC_BF16 : public testing::TestWithParam<std::vector<std::string>>
{
    MIOPEN_DECLARE_GTEST_USES_TEST_DRIVE();
};

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

void Run2dDriver(miopenDataType_t prec)
{
    std::vector<std::string> params;
    switch(prec)
    {
    case miopenFloat: params = GPU_PoolingBackward_NHWC_FP32::GetParam(); break;
    case miopenHalf: params = GPU_PoolingBackward_NHWC_FP16::GetParam(); break;
    case miopenBFloat16: params = GPU_PoolingBackward_NHWC_BF16::GetParam(); break;
    case miopenInt8:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
        FAIL() << "Data type not supported by pooling_backward_nhwc test";
    default: params = GPU_PoolingBackward_NHWC_FP32::GetParam();
    }

    for(const auto& test_value : params)
    {
        std::vector<std::string> tokens;
        GetArgs(test_value, tokens);
        std::vector<const char*> ptrs;

        std::transform(tokens.begin(), tokens.end(), std::back_inserter(ptrs), [](const auto& str) {
            return str.data();
        });

        testing::internal::CaptureStderr();
        test_drive<pooling2d_driver>(ptrs.size(), ptrs.data());
        auto capture = testing::internal::GetCapturedStderr();
        std::cerr << capture;
    }
}

bool IsTestSupportedForDevice() { return true; }

std::vector<std::string> GetTestCases(const std::string& precision)
{
    const auto& flag_arg = env::value(MIOPEN_TEST_FLAGS_ARGS);

    // Test backward pooling with NHWC layout
    // This triggers batched transpose (NHWC->NCHW before pooling, NCHW->NHWC after)
    const std::vector<std::string> test_cases = {
        // clang-format off
        {"test_pooling2d " + precision + " --forw 0 --in_layout NHWC --out_layout NHWC " + flag_arg}
        // clang-format on
    };

    return test_cases;
}

} // namespace pooling_backward_nhwc

using namespace pooling_backward_nhwc;

// FP32 backward pooling with NHWC layout
// This verifies: batched transpose selection + BF16 pooling backward + correct output
TEST_P(GPU_PoolingBackward_NHWC_FP32, FloatTest_pooling_backward_nhwc)
{
    if(IsTestSupportedForDevice())
    {
        Run2dDriver(miopenFloat);
    }
    else
    {
        GTEST_SKIP();
    }
}

// FP16 backward pooling with NHWC layout
TEST_P(GPU_PoolingBackward_NHWC_FP16, HalfTest_pooling_backward_nhwc)
{
    if(IsTestSupportedForDevice())
    {
        Run2dDriver(miopenHalf);
    }
    else
    {
        GTEST_SKIP();
    }
}

// BF16 backward pooling with NHWC layout
// This is the critical test: verifies end-to-end BF16 support through the stack
TEST_P(GPU_PoolingBackward_NHWC_BF16, BFloat16Test_pooling_backward_nhwc)
{
    if(IsTestSupportedForDevice())
    {
        Run2dDriver(miopenBFloat16);
    }
    else
    {
        GTEST_SKIP();
    }
}

// Smoke tests with minimal config
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_PoolingBackward_NHWC_FP32,
                         testing::Values(GetTestCases("--float")));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_PoolingBackward_NHWC_FP16,
                         testing::Values(GetTestCases("--half")));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_PoolingBackward_NHWC_BF16,
                         testing::Values(GetTestCases("--bfloat16")));
