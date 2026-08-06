// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "RMSNormTestCase.hpp"

namespace gpu_rmsnorm_ref_test
{

namespace
{

template <unsigned int BLOCK_SIZE>
int64_t getMaxGridSizeForCurrentDevice()
{
    static_assert(BLOCK_SIZE > 0, "BLOCK_SIZE must be greater than 0");

    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
    {
        return 1; // Default value
    }

    int deviceId = 0;
    if(hipGetDevice(&deviceId) != hipSuccess)
    {
        return 1;
    }

    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, deviceId) != hipSuccess)
    {
        return 1;
    }

    return static_cast<int64_t>(props.maxGridSize[0]) / static_cast<int64_t>(BLOCK_SIZE);
}

} // namespace

using hipdnn_data_sdk::utilities::TensorLayout;

// ============================================================================
// Small test cases for quick validation
// ============================================================================

inline std::vector<RMSNormTestCase> getRMSnormSmall4DTestCases()
{
    return {{{2, 3, 4, 4}, {1, 3, 4, 4}, TensorLayout::NCHW},
            {{32, 3, 8, 8}, {1, 1, 8, 8}, TensorLayout::NCHW},
            {{1, 3, 14, 14}, {1, 1, 1, 14}, TensorLayout::NCHW},
            {{16, 32, 24, 16}, {1, 32, 24, 16}, TensorLayout::NCHW},
            {{16, 40, 48, 32}, {1, 1, 48, 32}, TensorLayout::NCHW},

            {{2, 3, 4, 4}, {1, 3, 4, 4}, TensorLayout::NHWC},
            {{32, 3, 8, 8}, {1, 1, 8, 8}, TensorLayout::NHWC},
            {{1, 3, 14, 14}, {1, 1, 1, 14}, TensorLayout::NHWC},
            {{16, 32, 24, 16}, {1, 32, 24, 16}, TensorLayout::NHWC},
            {{16, 40, 48, 32}, {1, 1, 48, 32}, TensorLayout::NHWC}};
}

inline std::vector<RMSNormTestCase> getRMSnormSmall5DTestCases()
{
    return {{{2, 3, 3, 1, 1}, {1, 3, 3, 1, 1}, TensorLayout::NCDHW},
            {{2, 3, 4, 2, 2}, {1, 1, 4, 2, 2}, TensorLayout::NCDHW},
            {{2, 3, 4, 2, 2}, {1, 1, 1, 2, 2}, TensorLayout::NCDHW},
            {{2, 3, 4, 2, 2}, {1, 1, 1, 1, 2}, TensorLayout::NCDHW},
            {{4, 8, 2, 4, 4}, {1, 8, 2, 4, 4}, TensorLayout::NCDHW},

            {{2, 3, 3, 1, 1}, {1, 3, 3, 1, 1}, TensorLayout::NDHWC},
            {{2, 3, 4, 2, 2}, {1, 1, 4, 2, 2}, TensorLayout::NDHWC},
            {{2, 3, 4, 2, 2}, {1, 1, 1, 2, 2}, TensorLayout::NDHWC},
            {{2, 3, 4, 2, 2}, {1, 1, 1, 1, 2}, TensorLayout::NDHWC},
            {{4, 8, 2, 4, 4}, {1, 8, 2, 4, 4}, TensorLayout::NDHWC}};
}

// ============================================================================
// Medium test cases for standard validation
// ============================================================================

inline std::vector<RMSNormTestCase> getRMSnormMedium4DTestCases()
{
    return {{{1, 3, 14, 14}, {1, 3, 14, 14}, TensorLayout::NCHW},
            {{1, 3, 14, 14}, {1, 1, 14, 14}, TensorLayout::NCHW},
            {{1, 3, 14, 14}, {1, 1, 1, 14}, TensorLayout::NCHW},
            {{1, 256, 1, 1}, {1, 256, 1, 1}, TensorLayout::NCHW},
            {{2, 3, 1, 1}, {1, 3, 1, 1}, TensorLayout::NCHW},
            {{32, 1, 14, 14}, {1, 1, 14, 14}, TensorLayout::NCHW},
            {{32, 3, 1, 14}, {1, 3, 1, 14}, TensorLayout::NCHW},
            {{32, 3, 14, 1}, {1, 3, 14, 1}, TensorLayout::NCHW},
            {{32, 3, 14, 1}, {1, 1, 14, 1}, TensorLayout::NCHW},
            {{16, 32, 192, 128}, {1, 32, 192, 128}, TensorLayout::NCHW},
            {{16, 32, 192, 128}, {1, 1, 192, 128}, TensorLayout::NCHW},
            {{16, 64, 225, 225}, {1, 64, 225, 225}, TensorLayout::NCHW},
            {{16, 64, 225, 225}, {1, 1, 1, 225}, TensorLayout::NCHW},
            {{16, 128, 56, 56}, {1, 128, 56, 56}, TensorLayout::NCHW},
            {{16, 128, 56, 56}, {1, 1, 56, 56}, TensorLayout::NCHW},

            {{1, 3, 14, 14}, {1, 3, 14, 14}, TensorLayout::NHWC},
            {{1, 3, 14, 14}, {1, 1, 14, 14}, TensorLayout::NHWC},
            {{1, 3, 14, 14}, {1, 1, 1, 14}, TensorLayout::NHWC},
            {{1, 256, 1, 1}, {1, 256, 1, 1}, TensorLayout::NHWC},
            {{2, 3, 1, 1}, {1, 3, 1, 1}, TensorLayout::NHWC},
            {{32, 1, 14, 14}, {1, 1, 14, 14}, TensorLayout::NHWC},
            {{32, 3, 1, 14}, {1, 3, 1, 14}, TensorLayout::NHWC},
            {{32, 3, 14, 1}, {1, 3, 14, 1}, TensorLayout::NHWC},
            {{32, 3, 14, 1}, {1, 1, 14, 1}, TensorLayout::NHWC},
            {{16, 32, 192, 128}, {1, 32, 192, 128}, TensorLayout::NHWC},
            {{16, 32, 192, 128}, {1, 1, 192, 128}, TensorLayout::NHWC},
            {{16, 64, 225, 225}, {1, 64, 225, 225}, TensorLayout::NHWC},
            {{16, 64, 225, 225}, {1, 1, 1, 225}, TensorLayout::NHWC},
            {{16, 128, 56, 56}, {1, 128, 56, 56}, TensorLayout::NHWC},
            {{16, 128, 56, 56}, {1, 1, 56, 56}, TensorLayout::NHWC}};
}

inline std::vector<RMSNormTestCase> getRMSnormMedium5DTestCases()
{
    return {{{16, 3, 8, 14, 14}, {1, 3, 8, 14, 14}, TensorLayout::NCDHW},
            {{16, 32, 4, 48, 32}, {1, 32, 4, 48, 32}, TensorLayout::NCDHW},
            {{16, 32, 4, 48, 32}, {1, 1, 1, 48, 32}, TensorLayout::NCDHW},
            {{8, 64, 4, 28, 28}, {1, 64, 4, 28, 28}, TensorLayout::NCDHW},
            {{8, 64, 4, 28, 28}, {1, 1, 4, 28, 28}, TensorLayout::NCDHW},

            {{16, 3, 8, 14, 14}, {1, 3, 8, 14, 14}, TensorLayout::NDHWC},
            {{16, 32, 4, 48, 32}, {1, 32, 4, 48, 32}, TensorLayout::NDHWC},
            {{16, 32, 4, 48, 32}, {1, 1, 1, 48, 32}, TensorLayout::NDHWC},
            {{8, 64, 4, 28, 28}, {1, 64, 4, 28, 28}, TensorLayout::NDHWC},
            {{8, 64, 4, 28, 28}, {1, 1, 4, 28, 28}, TensorLayout::NDHWC}};
}

// ============================================================================
// Large test cases for comprehensive validation
// ============================================================================

inline std::vector<RMSNormTestCase> getRMSnormLarge4DTestCases()
{
    return {{{16, 288, 48, 32}, {1, 288, 48, 32}, TensorLayout::NCHW},
            {{16, 288, 48, 32}, {1, 1, 48, 32}, TensorLayout::NCHW},
            {{16, 576, 1, 30}, {1, 576, 1, 30}, TensorLayout::NCHW},
            {{16, 576, 1, 30}, {1, 1, 1, 30}, TensorLayout::NCHW},
            {{16, 2048, 16, 32}, {1, 2048, 16, 32}, TensorLayout::NCHW},
            {{16, 2048, 16, 32}, {1, 1, 16, 32}, TensorLayout::NCHW},
            {{128, 35, 48, 32}, {1, 35, 48, 32}, TensorLayout::NCHW},
            {{128, 512, 24, 48}, {1, 512, 24, 48}, TensorLayout::NCHW},
            {{128, 512, 24, 48}, {1, 1, 24, 48}, TensorLayout::NCHW},

            {{16, 288, 48, 32}, {1, 288, 48, 32}, TensorLayout::NHWC},
            {{16, 288, 48, 32}, {1, 1, 48, 32}, TensorLayout::NHWC},
            {{16, 576, 1, 30}, {1, 576, 1, 30}, TensorLayout::NHWC},
            {{16, 576, 1, 30}, {1, 1, 1, 30}, TensorLayout::NHWC},
            {{16, 2048, 16, 32}, {1, 2048, 16, 32}, TensorLayout::NHWC},
            {{16, 2048, 16, 32}, {1, 1, 16, 32}, TensorLayout::NHWC},
            {{128, 35, 48, 32}, {1, 35, 48, 32}, TensorLayout::NHWC},
            {{128, 512, 24, 48}, {1, 512, 24, 48}, TensorLayout::NHWC},
            {{128, 512, 24, 48}, {1, 1, 24, 48}, TensorLayout::NHWC}};
}

inline std::vector<RMSNormTestCase> getRMSnormLarge5DTestCases()
{
    return {{{16, 128, 8, 24, 16}, {1, 128, 8, 24, 16}, TensorLayout::NCDHW},
            {{16, 256, 4, 32, 32}, {1, 256, 4, 32, 32}, TensorLayout::NCDHW},
            {{16, 256, 4, 32, 32}, {1, 1, 4, 32, 32}, TensorLayout::NCDHW},
            {{32, 128, 8, 16, 16}, {1, 128, 8, 16, 16}, TensorLayout::NCDHW},
            {{32, 128, 8, 16, 16}, {1, 1, 1, 16, 16}, TensorLayout::NCDHW},

            {{16, 128, 8, 24, 16}, {1, 128, 8, 24, 16}, TensorLayout::NDHWC},
            {{16, 256, 4, 32, 32}, {1, 256, 4, 32, 32}, TensorLayout::NDHWC},
            {{16, 256, 4, 32, 32}, {1, 1, 4, 32, 32}, TensorLayout::NDHWC},
            {{32, 128, 8, 16, 16}, {1, 128, 8, 16, 16}, TensorLayout::NDHWC},
            {{32, 128, 8, 16, 16}, {1, 1, 1, 16, 16}, TensorLayout::NDHWC}};
}

// ============================================================================
// Edge-case shapes to isolate innerSize and outerSize growth.
// Only used with the DISABLED_TestGpuRMSNormFwdRefEdgeCaseValidation fixture.
// ============================================================================

inline std::vector<RMSNormTestCase> getRMSnormSkinnyModerateTestCases()
{
    // Matches the large scale dims used in getRMSnormLarge4DTestCases
    constexpr int64_t N = 2048;

    return {// 1 thread block, all work in one grid-stride loop
            {{1, 1, 1, N}, {1, 1, 1, N}, TensorLayout::NCHW},
            // Large thread blocks with a single active thread (degenerate reduction)
            {{N, 1, 1, 1}, {1, 1, 1, 1}, TensorLayout::NCHW},
            // Large thread blocks with two active threads (partial loop + two-thread reduction)
            {{N, 1, 1, 2}, {1, 1, 1, 2}, TensorLayout::NCHW}};
}

inline std::vector<RMSNormTestCase> getRMSnormSkinnyInt32ScaleTestCases(int64_t outerBound)
{
    // NOTE: INNER_BOUND in these cases should be INT32_MAX, but is reduced here due to
    // slow CPU fill/reference functions. Revisit once rocRAND-based GPU fill and
    // golden references for large tensors are available.
    constexpr int64_t INNER_BOUND = 100000000; // 100 million elements

    // Cases are same as getRMSnormSkinnyModerateTestCases, but with innerSize and outerSize set to the int32_t boundary values
    return {{{1, 1, 1, INNER_BOUND}, {1, 1, 1, INNER_BOUND}, TensorLayout::NCHW},
            {{outerBound, 1, 1, 1}, {1, 1, 1, 1}, TensorLayout::NCHW},
            {{outerBound, 1, 1, 2}, {1, 1, 1, 2}, TensorLayout::NCHW}};
}

inline std::vector<RMSNormTestCase> getRMSnormPowerOfTwoTestCases()
{
    // Outer size is fixed to small value to isolate innerSize's effect
    constexpr int64_t OUTER = 2;

    const std::vector<int64_t> innerSizes
        = {1,   2,   3,   7,   8,   9,   15,  16,  17,  63,   64,   65,
           127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025};

    std::vector<RMSNormTestCase> cases;
    cases.reserve(innerSizes.size());
    for(int64_t inner : innerSizes)
    {
        cases.push_back({{OUTER, 1, 1, inner}, {1, 1, 1, inner}, TensorLayout::NCHW});
    }
    return cases;
}

inline std::vector<RMSNormTestCase> getRMSnormInnerSizeInt32BoundaryTestCases()
{
    // Unguarded innerSize boundary test to verify int32 truncation is handled correctly
    // NOTE: INNER_BOUND in these cases should be INT32_MAX, but is reduced here due to
    // slow CPU fill/reference functions. Revisit once rocRAND-based GPU fill and
    // golden references for large tensors are available.
    constexpr int64_t INNER_BOUND = 100000000; // 100 million elements

    return {{{1, 1, 1, INNER_BOUND - 1}, {1, 1, 1, INNER_BOUND - 1}, TensorLayout::NCHW},
            {{1, 1, 1, INNER_BOUND}, {1, 1, 1, INNER_BOUND}, TensorLayout::NCHW},
            {{1, 1, 1, INNER_BOUND + 1}, {1, 1, 1, INNER_BOUND + 1}, TensorLayout::NCHW}};
}

} // namespace gpu_rmsnorm_ref_test
