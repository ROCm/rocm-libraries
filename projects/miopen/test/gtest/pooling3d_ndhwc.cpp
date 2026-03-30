// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <miopen/env.hpp>
#include "gtest_common.hpp"
#include "pooling3d_harness_gtest.hpp"

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLAGS_ARGS)

namespace env = miopen::env;

namespace pooling3d_ndhwc {

class GPU_Pooling3d_NDHWC_FP32 : public testing::TestWithParam<std::vector<std::string>>
{
    MIOPEN_DECLARE_GTEST_USES_TEST_DRIVE();
};

class GPU_Pooling3d_NDHWC_FP16 : public testing::TestWithParam<std::vector<std::string>>
{
    MIOPEN_DECLARE_GTEST_USES_TEST_DRIVE();
};

class GPU_Pooling3d_NDHWC_BFP16 : public testing::TestWithParam<std::vector<std::string>>
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

void Run3dHarness(miopenDataType_t prec)
{

    std::vector<std::string> params;
    switch(prec)
    {
    case miopenFloat: params = GPU_Pooling3d_NDHWC_FP32::GetParam(); break;
    case miopenHalf: params = GPU_Pooling3d_NDHWC_FP16::GetParam(); break;
    case miopenBFloat16: params = GPU_Pooling3d_NDHWC_BFP16::GetParam(); break;
    case miopenInt8:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
        FAIL() << "miopenInt8, miopenInt32, miopenDouble, miopenFloat8_fnuz, "
                  "miopenBFloat8_fnuz "
                  "data type not supported by "
                  "pooling3d_ndhwc test";

    default: params = GPU_Pooling3d_NDHWC_FP32::GetParam();
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
        test_drive<pooling3d_harness>(ptrs.size(), ptrs.data());
        auto capture = testing::internal::GetCapturedStderr();
        std::cerr << capture;
    }
};

bool IsTestSupportedForDevice() { return true; }

std::vector<std::string> GetTestCases(const std::string& precision)
{
    const auto& flag_arg = env::value(MIOPEN_TEST_FLAGS_ARGS);

    const std::vector<std::string> test_cases = {
        // clang-format off
        // Forward pooling with NDHWC layout (universal transpose - 3D)
        {"test_pooling3d " + precision + " --all --in_layout NDHWC --out_layout NDHWC " + flag_arg},
        // Backward pooling with NDHWC layout (universal transpose - 3D)
        {"test_pooling3d " + precision + " --forw 0 --in_layout NDHWC --out_layout NDHWC " + flag_arg}
        // clang-format on
    };

    return test_cases;
}

} // namespace pooling3d_ndhwc
using namespace pooling3d_ndhwc;

TEST_P(GPU_Pooling3d_NDHWC_FP32, FloatTest_pooling3d_ndhwc)
{
    if(IsTestSupportedForDevice())
    {
        Run3dHarness(miopenFloat);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(GPU_Pooling3d_NDHWC_FP16, HalfTest_pooling3d_ndhwc)
{
    if(IsTestSupportedForDevice())
    {
        Run3dHarness(miopenHalf);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(GPU_Pooling3d_NDHWC_BFP16, BFloat16Test_pooling3d_ndhwc)
{
    if(IsTestSupportedForDevice())
    {
        Run3dHarness(miopenBFloat16);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(Full, GPU_Pooling3d_NDHWC_FP32, testing::Values(GetTestCases("--float")));

INSTANTIATE_TEST_SUITE_P(Full, GPU_Pooling3d_NDHWC_FP16, testing::Values(GetTestCases("--half")));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Pooling3d_NDHWC_BFP16,
                         testing::Values(GetTestCases("--bfloat16")));
