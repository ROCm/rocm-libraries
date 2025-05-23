/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <tuple>
#include <string_view>

#include "gtest_common.hpp"
#include "get_handle.hpp"

#include "../conv2d.hpp"

namespace {

auto GetTestCases()
{
    // MIOPEN_DEBUG_TUNING_ITERATIONS_MAX is set to 2 because kernels are very slow to build.
    // MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW is explicitly enabled due to the kernel is
    // disabled by default via #2306
    const auto env_fwd =
        std::tuple{std::pair{MIOPEN_FIND_ENFORCE, "SEARCH_DB_UPDATE"},
                   std::pair{wa::MIOPEN_DEBUG_TUNING_ITERATIONS_MAX, 2},
                   std::pair{wa::MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL, 0},
                   std::pair{MIOPEN_FIND_MODE, "normal"},
                   std::pair{MIOPEN_DEBUG_FIND_ONLY_SOLVER, "ConvCkIgemmFwdV6r1DlopsNchw"},
                   std::pair{MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW, true}};

    const std::string vf = " --verbose --disable-backward-data --disable-backward-weights";

    return std::vector{
        // clang-format off
    std::pair{env_fwd, vf + " --input 128 64 56 56 --weights 256 64 1 1 --pads_strides_dilations 0 0 1 1 1 1"}
        // clang-format on
    };
}

using TestCase = decltype(GetTestCases())::value_type;

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::gfx103X>;
    using d_mask = disabled<Gpu::Default>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

} // namespace

class GPU_Conv2dTuningV6R1_FP16 : public HalfTestCase<std::vector<TestCase>>
{
};

TEST_P(GPU_Conv2dTuningV6R1_FP16, HalfTest_smoke_solver_ConvCkIgemmFwdV6r1DlopsNchw)
{
    if(IsTestSupportedForDevice())
    {
        invoke_with_params<conv2d_driver, GPU_Conv2dTuningV6R1_FP16>(tuning_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Conv2dTuningV6R1_FP16, testing::Values(GetTestCases()));
