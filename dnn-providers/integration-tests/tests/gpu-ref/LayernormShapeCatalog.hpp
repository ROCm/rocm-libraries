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

// P1 (dtype orthogonality): the datatype only enters through per-element load/store conversion,
// so reduced-precision (Standard tier) runs need only one case per layout/optional-tensor state.
// Every distinct shape/layout is a separate ~0.45 s HipRTC compile (~0.95 s for bwd).
inline std::vector<LayernormTestCase> getLayernormSmall4DDtypeTestCases()
{
    return {
        {{2, 2, 3, 2}, 2, true, TensorLayout::NCHW},
        {{2, 2, 3, 2}, 2, false, TensorLayout::NHWC},
    };
}

inline std::vector<LayernormTestCase> getLayernormSmall5DDtypeTestCases()
{
    return {
        {{2, 2, 3, 2, 2}, 3, true, TensorLayout::NCDHW},
        {{2, 2, 3, 2, 2}, 3, false, TensorLayout::NDHWC},
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
    // Imported from MIOpen, culled (P3). These cases normalize only the last dimension, so
    // INNER_SIZE is that dimension (< LOCAL_SIZE) and the kernel loop runs once for every shape;
    // shapes differ only in OUTER_SIZE and, for NDHWC with C > 1, STRIDE = C. Each distinct
    // shape is a separate ~0.45 s HipRTC compile.
    //   {32,1,14,14,14}: C=1 (NDHWC memory-identical to NCDHW)
    //   {32,32,6,6,6} and {32,32,6,4,12}: C=32 (NDHWC STRIDE 32)
    return {
        {{32, 1, 14, 14, 14}, 4, false, TensorLayout::NCDHW},
        {{32, 32, 6, 6, 6}, 4, false, TensorLayout::NCDHW},
        {{32, 32, 6, 4, 12}, 4, false, TensorLayout::NCDHW},
        {{32, 1, 14, 14, 14}, 4, true, TensorLayout::NCDHW},
        {{32, 32, 6, 6, 6}, 4, true, TensorLayout::NCDHW},
        {{32, 32, 6, 4, 12}, 4, true, TensorLayout::NCDHW},
        {{32, 1, 14, 14, 14}, 4, false, TensorLayout::NDHWC},
        {{32, 32, 6, 6, 6}, 4, false, TensorLayout::NDHWC},
        {{32, 32, 6, 4, 12}, 4, false, TensorLayout::NDHWC},
        {{32, 1, 14, 14, 14}, 4, true, TensorLayout::NDHWC},
        {{32, 32, 6, 6, 6}, 4, true, TensorLayout::NDHWC},
        {{32, 32, 6, 4, 12}, 4, true, TensorLayout::NDHWC},
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
    // Imported from MIOpen, culled (P3). As for the medium catalog, only the last dimension is
    // normalized, so each shape runs the same single-iteration loop and differs only in
    // OUTER_SIZE and STRIDE; each distinct shape is a separate HipRTC compile.
    //   {32,2,32,57,125} and {1,3,16,240,320}: C>1 (NDHWC STRIDE = C)
    //   {256,1,32,32,32}: C=1 (NDHWC memory-identical to NCDHW)
    return {
        {{32, 2, 32, 57, 125}, 4, false, TensorLayout::NCDHW},
        {{1, 3, 16, 240, 320}, 4, false, TensorLayout::NCDHW},
        {{256, 1, 32, 32, 32}, 4, false, TensorLayout::NCDHW},
        {{32, 2, 32, 57, 125}, 4, true, TensorLayout::NCDHW},
        {{1, 3, 16, 240, 320}, 4, true, TensorLayout::NCDHW},
        {{256, 1, 32, 32, 32}, 4, true, TensorLayout::NCDHW},
        {{32, 2, 32, 57, 125}, 4, false, TensorLayout::NDHWC},
        {{1, 3, 16, 240, 320}, 4, false, TensorLayout::NDHWC},
        {{256, 1, 32, 32, 32}, 4, false, TensorLayout::NDHWC},
        {{32, 2, 32, 57, 125}, 4, true, TensorLayout::NDHWC},
        {{1, 3, 16, 240, 320}, 4, true, TensorLayout::NDHWC},
        {{256, 1, 32, 32, 32}, 4, true, TensorLayout::NDHWC},
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
