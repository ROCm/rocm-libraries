// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <tuple>
#include <string_view>

#include "gtest_common.hpp"

#include "conv2d.hpp"

namespace {

auto GetTestCases()
{
    const auto env_fwd = std::tuple{
        std::pair{MIOPEN_FIND_ENFORCE, "SEARCH_DB_UPDATE"},
        std::pair{wa::MIOPEN_DEBUG_TUNING_ITERATIONS_MAX, 5},
        std::pair{MIOPEN_FIND_MODE, "normal"},
        std::pair{MIOPEN_DEBUG_FIND_ONLY_SOLVER, "ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC"}};

    const std::string vf = " --verbose --disable-backward-data --disable-backward-weights";
    const std::string layout =
        " --in_layout NCHW --fil_layout CHWN --out_layout NCHW --tensor_vect 1 --vector_length 4";

    return std::vector{
        // clang-format off
    std::pair{env_fwd, vf + " --input 64 256 7 7 --weights 256 3 3 128 --pads_strides_dilations 0 0 1 1 1 1" + layout}
        // clang-format on
    };
}

using TestCase = decltype(GetTestCases())::value_type;

bool SkipTest() { return get_handle_xnack(); }

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::gfx103X>;
    using d_mask = disabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx908, Gpu::gfx90A>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

} // namespace

class GPU_Conv2dTuningDynamicFwdDlops_FP16 : public HalfTestCase<std::vector<TestCase>>
{
    MIOPEN_DECLARE_GTEST_USES_TEST_DRIVE();
};

TEST_P(GPU_Conv2dTuningDynamicFwdDlops_FP16,
       HalfTest_smoke_solver_ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC)
{
    if(IsTestSupportedForDevice() && !SkipTest())
    {
        invoke_with_params<conv2d_driver, GPU_Conv2dTuningDynamicFwdDlops_FP16>(tuning_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv2dTuningDynamicFwdDlops_FP16,
                         testing::Values(GetTestCases()));
