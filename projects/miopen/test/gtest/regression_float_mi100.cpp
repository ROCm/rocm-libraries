// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <tuple>
#include <string_view>

#include "gtest_common.hpp"
#include "get_handle.hpp"

#include "conv2d.hpp"

namespace regression_float_mi100 {

auto GetTestCases()
{
    // Regression test for SWDEV-305815 (issue 1206)
    const auto env = std::tuple{std::pair{MIOPEN_DEBUG_CONV_WINOGRAD, false},
                                std::pair{MIOPEN_DEBUG_CONV_FFT, false},
                                std::pair{MIOPEN_DEBUG_CONV_DIRECT, false},
                                std::pair{MIOPEN_DEBUG_CONV_GEMM, false},
                                std::pair{MIOPEN_DEBUG_CONV_IMPLICIT_GEMM, false},
                                std::pair{MIOPEN_LOG_LEVEL, 1}};

    const std::string v          = " --verbose";
    const std::string dis_fwd    = " --disable-forward";
    const std::string dis_bk_wei = " --disable-backward-weights";

    return std::vector{
        // clang-format off
    std::pair{env, v + " --input 32 256 38 38 --weights 256 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei}
        // clang-format on
    };
}

using TestCase = decltype(GetTestCases())::value_type;

class GPU_Conv2d_regression_mi100_FP32 : public FloatTestCase<std::vector<TestCase>>
{
    MIOPEN_DECLARE_GTEST_USES_TEST_DRIVE();
};

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::Default>;
    using d_mask = disabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx90A>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

} // namespace regression_float_mi100
using namespace regression_float_mi100;

TEST_P(GPU_Conv2d_regression_mi100_FP32, FloatTest)
{
    if(IsTestSupportedForDevice())
    {
        invoke_with_params<conv2d_driver, GPU_Conv2d_regression_mi100_FP32>(default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Conv2d_regression_mi100_FP32, testing::Values(GetTestCases()));
