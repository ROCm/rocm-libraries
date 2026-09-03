// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "LayernormTestCase.hpp"

namespace gpu_layernorm_ref_test
{

using hipdnn_data_sdk::utilities::TensorLayout;

inline std::vector<LayernormTestCase> getLayernormSmall4DTestCases()
{
    return {
        {{2, 2, 3, 2}, 3, false, TensorLayout::NCHW}, // Minimal test cases
        {{2, 2, 3, 2}, 2, false, TensorLayout::NCHW},
        {{2, 2, 3, 2}, 1, false, TensorLayout::NCHW},
        {{2, 2, 3, 2}, 3, true, TensorLayout::NCHW},
        {{2, 2, 3, 2}, 2, true, TensorLayout::NCHW},
        {{2, 2, 3, 2}, 1, true, TensorLayout::NCHW},
        {{2, 5, 2, 2}, 1, true, TensorLayout::NCHW}, // Edge case: larger C with normalized dim 1
        {{2, 2, 3, 2}, 3, false, TensorLayout::NHWC}, // Minimal test cases
        {{2, 2, 3, 2}, 2, false, TensorLayout::NHWC},
        {{2, 2, 3, 2}, 1, false, TensorLayout::NHWC},
        {{2, 2, 3, 2}, 3, true, TensorLayout::NHWC},
        {{2, 2, 3, 2}, 2, true, TensorLayout::NHWC},
        {{2, 2, 3, 2}, 1, true, TensorLayout::NHWC},
        {{2, 5, 2, 2}, 1, true, TensorLayout::NHWC}, // Edge case: larger C with normalized dim 1
    };
}

inline std::vector<LayernormTestCase> getLayernormSmall5DTestCases()
{
    return {
        {{2, 2, 3, 2, 2}, 4, false, TensorLayout::NCDHW}, // Minimal test cases
        {{2, 2, 3, 2, 2}, 3, false, TensorLayout::NCDHW},
        {{2, 2, 3, 2, 2}, 2, false, TensorLayout::NCDHW},
        {{2, 2, 3, 2, 2}, 1, false, TensorLayout::NCDHW},
        {{2, 2, 3, 2, 2}, 4, true, TensorLayout::NCDHW},
        {{2, 2, 3, 2, 2}, 3, true, TensorLayout::NCDHW},
        {{2, 2, 3, 2, 2}, 2, true, TensorLayout::NCDHW},
        {{2, 2, 3, 2, 2}, 1, true, TensorLayout::NCDHW},
        {{2, 5, 2, 2, 2},
         1,
         true,
         TensorLayout::NCDHW}, // Edge case: larger C with normalized dim 1
        {{2, 2, 3, 2, 2}, 4, false, TensorLayout::NDHWC}, // Minimal test cases
        {{2, 2, 3, 2, 2}, 3, false, TensorLayout::NDHWC},
        {{2, 2, 3, 2, 2}, 2, false, TensorLayout::NDHWC},
        {{2, 2, 3, 2, 2}, 1, false, TensorLayout::NDHWC},
        {{2, 2, 3, 2, 2}, 4, true, TensorLayout::NDHWC},
        {{2, 2, 3, 2, 2}, 3, true, TensorLayout::NDHWC},
        {{2, 2, 3, 2, 2}, 2, true, TensorLayout::NDHWC},
        {{2, 2, 3, 2, 2}, 1, true, TensorLayout::NDHWC},
        {{2, 5, 2, 2, 2},
         1,
         true,
         TensorLayout::NDHWC}, // Edge case: larger C with normalized dim 1
    };
}

inline std::vector<LayernormTestCase> getLayernormMedium4DTestCases()
{
    // Imported from MIOpen
    return {
        {{32, 4, 4, 256}, 1, false, TensorLayout::NCHW},
        {{32, 4, 4, 256}, 1, true, TensorLayout::NCHW},
        {{32, 4, 4, 256}, 1, false, TensorLayout::NHWC},
        {{32, 4, 4, 256}, 1, true, TensorLayout::NHWC},
    };
}

inline std::vector<LayernormTestCase> getLayernormMedium5DTestCases()
{
    // Imported from MIOpen
    return {
        {{32, 1, 32, 32, 32}, 4, false, TensorLayout::NCDHW}, // 32x32x32 based on VoxNet arch
        {{32, 1, 14, 14, 14}, 4, false, TensorLayout::NCDHW},
        {{32, 32, 14, 14, 14}, 4, false, TensorLayout::NCDHW},
        {{32, 32, 12, 12, 12}, 4, false, TensorLayout::NCDHW},
        {{32, 32, 6, 6, 6}, 4, false, TensorLayout::NCDHW},
        {{32, 32, 14, 25, 59}, 4, false, TensorLayout::NCDHW},
        {{32, 32, 6, 10, 27}, 4, false, TensorLayout::NCDHW},
        {{32, 32, 4, 6, 11}, 4, false, TensorLayout::NCDHW},
        {{32, 32, 2, 2, 3}, 4, false, TensorLayout::NCDHW},
        {{32, 32, 32, 28, 62},
         4,
         false,
         TensorLayout::NCDHW}, // Hand-gesture recognition CVPR 2015 paper Low Res Net Path
        {{32, 32, 14, 12, 29}, 4, false, TensorLayout::NCDHW},
        {{32, 32, 6, 4, 12}, 4, false, TensorLayout::NCDHW},
        {{32, 32, 4, 2, 2}, 4, false, TensorLayout::NCDHW},
        {{16, 32, 6, 50, 50}, 4, false, TensorLayout::NCDHW}, // Multi-view 3D convnet
        {{32, 1, 32, 32, 32}, 4, true, TensorLayout::NCDHW}, // 32x32x32 based on VoxNet arch
        {{32, 1, 14, 14, 14}, 4, true, TensorLayout::NCDHW},
        {{32, 32, 14, 14, 14}, 4, true, TensorLayout::NCDHW},
        {{32, 32, 12, 12, 12}, 4, true, TensorLayout::NCDHW},
        {{32, 32, 6, 6, 6}, 4, true, TensorLayout::NCDHW},
        {{32, 32, 14, 25, 59}, 4, true, TensorLayout::NCDHW},
        {{32, 32, 6, 10, 27}, 4, true, TensorLayout::NCDHW},
        {{32, 32, 4, 6, 11}, 4, true, TensorLayout::NCDHW},
        {{32, 32, 2, 2, 3}, 4, true, TensorLayout::NCDHW},
        {{32, 32, 32, 28, 62},
         4,
         true,
         TensorLayout::NCDHW}, // Hand-gesture recognition CVPR 2015 paper Low Res Net Path
        {{32, 32, 14, 12, 29}, 4, true, TensorLayout::NCDHW},
        {{32, 32, 6, 4, 12}, 4, true, TensorLayout::NCDHW},
        {{32, 32, 4, 2, 2}, 4, true, TensorLayout::NCDHW},
        {{16, 32, 6, 50, 50}, 4, true, TensorLayout::NCDHW}, // Multi-view 3D convnet
        {{32, 1, 32, 32, 32}, 4, false, TensorLayout::NDHWC}, // 32x32x32 based on VoxNet arch
        {{32, 1, 14, 14, 14}, 4, false, TensorLayout::NDHWC},
        {{32, 32, 14, 14, 14}, 4, false, TensorLayout::NDHWC},
        {{32, 32, 12, 12, 12}, 4, false, TensorLayout::NDHWC},
        {{32, 32, 6, 6, 6}, 4, false, TensorLayout::NDHWC},
        {{32, 32, 14, 25, 59}, 4, false, TensorLayout::NDHWC},
        {{32, 32, 6, 10, 27}, 4, false, TensorLayout::NDHWC},
        {{32, 32, 4, 6, 11}, 4, false, TensorLayout::NDHWC},
        {{32, 32, 2, 2, 3}, 4, false, TensorLayout::NDHWC},
        {{32, 32, 32, 28, 62},
         4,
         false,
         TensorLayout::NDHWC}, // Hand-gesture recognition CVPR 2015 paper Low Res Net Path
        {{32, 32, 14, 12, 29}, 4, false, TensorLayout::NDHWC},
        {{32, 32, 6, 4, 12}, 4, false, TensorLayout::NDHWC},
        {{32, 32, 4, 2, 2}, 4, false, TensorLayout::NDHWC},
        {{16, 32, 6, 50, 50}, 4, false, TensorLayout::NDHWC}, // Multi-view 3D convnet
        {{32, 1, 32, 32, 32}, 4, true, TensorLayout::NDHWC}, // 32x32x32 based on VoxNet arch
        {{32, 1, 14, 14, 14}, 4, true, TensorLayout::NDHWC},
        {{32, 32, 14, 14, 14}, 4, true, TensorLayout::NDHWC},
        {{32, 32, 12, 12, 12}, 4, true, TensorLayout::NDHWC},
        {{32, 32, 6, 6, 6}, 4, true, TensorLayout::NDHWC},
        {{32, 32, 14, 25, 59}, 4, true, TensorLayout::NDHWC},
        {{32, 32, 6, 10, 27}, 4, true, TensorLayout::NDHWC},
        {{32, 32, 4, 6, 11}, 4, true, TensorLayout::NDHWC},
        {{32, 32, 2, 2, 3}, 4, true, TensorLayout::NDHWC},
        {{32, 32, 32, 28, 62},
         4,
         true,
         TensorLayout::NDHWC}, // Hand-gesture recognition CVPR 2015 paper Low Res Net Path
        {{32, 32, 14, 12, 29}, 4, true, TensorLayout::NDHWC},
        {{32, 32, 6, 4, 12}, 4, true, TensorLayout::NDHWC},
        {{32, 32, 4, 2, 2}, 4, true, TensorLayout::NDHWC},
        {{16, 32, 6, 50, 50}, 4, true, TensorLayout::NDHWC}, // Multi-view 3D convnet
    };
}

inline std::vector<LayernormTestCase> getLayernormLarge4DTestCases()
{
    // Imported from MIOpen
    return {
        {{64, 4, 4, 256}, 1, false, TensorLayout::NCHW},
        {{64, 4, 4, 256}, 1, true, TensorLayout::NCHW},
        {{64, 4, 4, 256}, 1, false, TensorLayout::NHWC},
        {{64, 4, 4, 256}, 1, true, TensorLayout::NHWC},
    };
}

inline std::vector<LayernormTestCase> getLayernormLarge5DTestCases()
{
    // Imported from MIOpen
    return {
        {{256, 1, 32, 32, 32}, 4, false, TensorLayout::NCDHW}, // 32x32x32 based on VoxNet arch
        {{256, 32, 14, 14, 14}, 4, false, TensorLayout::NCDHW},
        {{256, 32, 12, 12, 12}, 4, false, TensorLayout::NCDHW},
        {{256, 32, 6, 6, 6}, 4, false, TensorLayout::NCDHW},
        {{512, 1, 32, 32, 32}, 4, false, TensorLayout::NCDHW}, // 32x32x32 based on VoxNet arch
        {{512, 32, 14, 14, 14}, 4, false, TensorLayout::NCDHW},
        {{512, 32, 12, 12, 12}, 4, false, TensorLayout::NCDHW},
        {{512, 32, 6, 6, 6}, 4, false, TensorLayout::NCDHW},
        {{32, 2, 32, 57, 125},
         4,
         false,
         TensorLayout::NCDHW}, // Hand-gesture recognition CVPR 2015 paper High Res Net Path
        {{1, 3, 8, 240, 320}, 4, false, TensorLayout::NCDHW}, // 3D convet on video
        {{1, 3, 16, 240, 320}, 4, false, TensorLayout::NCDHW}, // 3D convet on video
        {{1, 3, 8, 128, 171}, 4, false, TensorLayout::NCDHW}, // 3D convet on video
        {{1, 3, 16, 128, 171}, 4, false, TensorLayout::NCDHW}, // 3D convet on video
        {{1, 3, 8, 112, 112}, 4, false, TensorLayout::NCDHW}, // 3D convet on video
        {{1, 3, 16, 112, 112}, 4, false, TensorLayout::NCDHW}, // 3D convet on video
        {{256, 1, 32, 32, 32}, 4, true, TensorLayout::NCDHW}, // 32x32x32 based on VoxNet arch
        {{256, 32, 14, 14, 14}, 4, true, TensorLayout::NCDHW},
        {{256, 32, 12, 12, 12}, 4, true, TensorLayout::NCDHW},
        {{256, 32, 6, 6, 6}, 4, true, TensorLayout::NCDHW},
        {{512, 1, 32, 32, 32}, 4, true, TensorLayout::NCDHW}, // 32x32x32 based on VoxNet arch
        {{512, 32, 14, 14, 14}, 4, true, TensorLayout::NCDHW},
        {{512, 32, 12, 12, 12}, 4, true, TensorLayout::NCDHW},
        {{512, 32, 6, 6, 6}, 4, true, TensorLayout::NCDHW},
        {{32, 2, 32, 57, 125},
         4,
         true,
         TensorLayout::NCDHW}, // Hand-gesture recognition CVPR 2015 paper High Res Net Path
        {{1, 3, 8, 240, 320}, 4, true, TensorLayout::NCDHW}, // 3D convet on video
        {{1, 3, 16, 240, 320}, 4, true, TensorLayout::NCDHW}, // 3D convet on video
        {{1, 3, 8, 128, 171}, 4, true, TensorLayout::NCDHW}, // 3D convet on video
        {{1, 3, 16, 128, 171}, 4, true, TensorLayout::NCDHW}, // 3D convet on video
        {{1, 3, 8, 112, 112}, 4, true, TensorLayout::NCDHW}, // 3D convet on video
        {{1, 3, 16, 112, 112}, 4, true, TensorLayout::NCDHW}, // 3D convet on video
        {{256, 1, 32, 32, 32}, 4, false, TensorLayout::NDHWC}, // 32x32x32 based on VoxNet arch
        {{256, 32, 14, 14, 14}, 4, false, TensorLayout::NDHWC},
        {{256, 32, 12, 12, 12}, 4, false, TensorLayout::NDHWC},
        {{256, 32, 6, 6, 6}, 4, false, TensorLayout::NDHWC},
        {{512, 1, 32, 32, 32}, 4, false, TensorLayout::NDHWC}, // 32x32x32 based on VoxNet arch
        {{512, 32, 14, 14, 14}, 4, false, TensorLayout::NDHWC},
        {{512, 32, 12, 12, 12}, 4, false, TensorLayout::NDHWC},
        {{512, 32, 6, 6, 6}, 4, false, TensorLayout::NDHWC},
        {{32, 2, 32, 57, 125},
         4,
         false,
         TensorLayout::NDHWC}, // Hand-gesture recognition CVPR 2015 paper High Res Net Path
        {{1, 3, 8, 240, 320}, 4, false, TensorLayout::NDHWC}, // 3D convet on video
        {{1, 3, 16, 240, 320}, 4, false, TensorLayout::NDHWC}, // 3D convet on video
        {{1, 3, 8, 128, 171}, 4, false, TensorLayout::NDHWC}, // 3D convet on video
        {{1, 3, 16, 128, 171}, 4, false, TensorLayout::NDHWC}, // 3D convet on video
        {{1, 3, 8, 112, 112}, 4, false, TensorLayout::NDHWC}, // 3D convet on video
        {{1, 3, 16, 112, 112}, 4, false, TensorLayout::NDHWC}, // 3D convet on video
        {{256, 1, 32, 32, 32}, 4, true, TensorLayout::NDHWC}, // 32x32x32 based on VoxNet arch
        {{256, 32, 14, 14, 14}, 4, true, TensorLayout::NDHWC},
        {{256, 32, 12, 12, 12}, 4, true, TensorLayout::NDHWC},
        {{256, 32, 6, 6, 6}, 4, true, TensorLayout::NDHWC},
        {{512, 1, 32, 32, 32}, 4, true, TensorLayout::NDHWC}, // 32x32x32 based on VoxNet arch
        {{512, 32, 14, 14, 14}, 4, true, TensorLayout::NDHWC},
        {{512, 32, 12, 12, 12}, 4, true, TensorLayout::NDHWC},
        {{512, 32, 6, 6, 6}, 4, true, TensorLayout::NDHWC},
        {{32, 2, 32, 57, 125},
         4,
         true,
         TensorLayout::NDHWC}, // Hand-gesture recognition CVPR 2015 paper High Res Net Path
        {{1, 3, 8, 240, 320}, 4, true, TensorLayout::NDHWC}, // 3D convet on video
        {{1, 3, 16, 240, 320}, 4, true, TensorLayout::NDHWC}, // 3D convet on video
        {{1, 3, 8, 128, 171}, 4, true, TensorLayout::NDHWC}, // 3D convet on video
        {{1, 3, 16, 128, 171}, 4, true, TensorLayout::NDHWC}, // 3D convet on video
        {{1, 3, 8, 112, 112}, 4, true, TensorLayout::NDHWC}, // 3D convet on video
        {{1, 3, 16, 112, 112}, 4, true, TensorLayout::NDHWC}, // 3D convet on video
    };
}

// ============================================================================
// Edge-case shapes to isolate innerSize and outerSize growth.
// Only used with the DISABLED_TestGpuLayernormFwdRefEdgeCaseValidation fixture.
// ============================================================================

inline std::vector<LayernormTestCase> getLayernormSkinnyModerateTestCases()
{
    // Matches the large scale dims used in getLayernormLarge4DTestCases
    constexpr int64_t N = 2048;

    return {// 1 thread block, all work in one grid-stride loop
            {{1, 1, 1, N}, 3, true, TensorLayout::NCHW},
            // Large thread blocks with a single active thread (degenerate reduction)
            {{N, 1, 1, 1}, 3, true, TensorLayout::NCHW},
            // Large thread blocks with two active threads (partial loop + two-thread reduction)
            {{N, 1, 1, 2}, 3, true, TensorLayout::NCHW}};
}

inline std::vector<LayernormTestCase> getLayernormSkinnyInt32ScaleTestCases(int64_t outerBound)
{
    // NOTE: INNER_BOUND in these cases should be INT32_MAX, but is reduced here due to
    // slow CPU fill/reference functions. Revisit once rocRAND-based GPU fill and
    // golden references for large tensors are available.
    constexpr int64_t INNER_BOUND = 100000000; // 100 million elements

    // Cases are same as getLayernormSkinnyModerateTestCases, but with innerSize and outerSize set to the int32_t boundary values
    return {{{1, 1, 1, INNER_BOUND}, 3, true, TensorLayout::NCHW},
            {{outerBound, 1, 1, 1}, 3, true, TensorLayout::NCHW},
            {{outerBound, 1, 1, 2}, 3, true, TensorLayout::NCHW}};
}

inline std::vector<LayernormTestCase> getLayernormPowerOfTwoTestCases()
{
    // Outer size is fixed to small value to isolate innerSize's effect
    constexpr int64_t OUTER = 2;

    const std::vector<int64_t> innerSizes
        = {1,   2,   3,   7,   8,   9,   15,  16,  17,  63,   64,   65,
           127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025};

    std::vector<LayernormTestCase> cases;
    cases.reserve(innerSizes.size());
    for(int64_t inner : innerSizes)
    {
        cases.push_back({{OUTER, 1, 1, inner}, 3, true, TensorLayout::NCHW});
    }
    return cases;
}

inline std::vector<LayernormTestCase> getLayernormInnerSizeInt32BoundaryTestCases()
{
    // Unguarded innerSize boundary test to verify int32 truncation is handled correctly
    // NOTE: INNER_BOUND in these cases should be INT32_MAX, but is reduced here due to
    // slow CPU fill/reference functions. Revisit once rocRAND-based GPU fill and
    // golden references for large tensors are available.
    constexpr int64_t INNER_BOUND = 100000000; // 100 million elements

    return {{{1, 1, 1, INNER_BOUND - 1}, 3, true, TensorLayout::NCHW},
            {{1, 1, 1, INNER_BOUND}, 3, true, TensorLayout::NCHW},
            {{1, 1, 1, INNER_BOUND + 1}, 3, true, TensorLayout::NCHW}};
}

} // namespace gpu_layernorm_ref_test
