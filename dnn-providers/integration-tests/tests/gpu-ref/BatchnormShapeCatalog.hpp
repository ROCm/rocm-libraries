// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "BatchnormTestCase.hpp"

namespace gpu_batchnorm_ref_test
{

using hipdnn_data_sdk::utilities::TensorLayout;

// ============================================================================
// Small test cases for quick validation
// ============================================================================

inline std::vector<BatchnormTestCase> getBatchnormSmall3DTestCases()
{
    return {
        {{2, 3, 4}}, // 24
        {{4, 3, 1}}, // 12
        {{1, 111, 1}} // 111
    };
}

inline std::vector<BatchnormTestCase> getBatchnormSmall4DTestCases()
{
    return {
        {{2, 3, 1, 1}}, // 6
        {{1, 256, 1, 1}}, // 256
        {{1, 3, 14, 14}}, // 588
        {{32, 1, 14, 14}}, // 6272
        {{32, 14, 1, 14}}, // 6272
        {{32, 14, 14, 1}}, // 6272

    };
}

inline std::vector<BatchnormTestCase> getBatchnormSmall5DTestCases()
{
    return {
        {{2, 3, 3, 1, 1}}, // 18
        {{1, 128, 1, 1, 1}}, // 128
        {{256, 1, 2, 2, 2}} // 2048
    };
}

// ============================================================================
// Medium shapes — Standard tier (PR gate)
// ============================================================================
inline std::vector<BatchnormTestCase> getBatchnormMedium3DTestCases()
{
    return {
        {{1, 64, 1024}}, // 65536
        {{2, 16, 512}}, // 16384
        {{2, 1, 2048}} // 4096

    };
}

inline std::vector<BatchnormTestCase> getBatchnormMedium4DTestCases()
{
    return {
        {{16, 32, 4, 8}}, // 16384
        {{2, 1, 225, 225}}, // 101250
        {{3, 1, 127, 127}}, // 48387
        {{1, 65536, 1, 1}} // 65536
    };
}

inline std::vector<BatchnormTestCase> getBatchnormMedium5DTestCases()
{
    return {
        {{16, 3, 8, 14, 14}}, // 75264
        {{3, 32, 4, 7, 32}}, // 86016
        {{1, 30, 28, 28, 4}}, // 94080
        {{256, 1, 8, 8, 8}}, // 131072

    };
}

// ============================================================================
// Large edge test cases for comprehensive validation
// ============================================================================

inline std::vector<BatchnormTestCase> getBatchnormLargeEdge3DTestCases()
{
    return {
        {{1, 2048, 256}}, // 524288
        {{16, 1, 65536}}, // 1048576
        {{4096, 1, 333}} //  1363968
    };
}

inline std::vector<BatchnormTestCase> getBatchnormLargeEdge4DTestCases()
{
    return {
        {{1, 1, 2660, 2660}}, // 7075600
        {{1, 276480, 1, 1}}, // 276480
        {{16777216, 1, 1, 1}}, // 16777216
        {{1, 4480, 48, 32}}, // 6881280
        {{56, 1, 128, 128}}, // 917504
        {{256, 2048, 1, 1}} // 524288
    };
}

inline std::vector<BatchnormTestCase> getBatchnormLargeEdge5DTestCases()
{
    return {
        {{6291456, 1, 1, 1, 1}}, // 6291456
        {{1, 8388608, 1, 1, 1}}, // 8388608
        {{1, 1, 160, 160, 160}} // 4096000
    };
}

// ============================================================================
// Large stress test cases for full validation
// ============================================================================

inline std::vector<BatchnormTestCase> getBatchnormLargeStress3DTestCases()
{
    return {
        {{8, 32, 2048}}, // 524288
        {{16, 128, 512}}, // 1048576
    };
}

inline std::vector<BatchnormTestCase> getBatchnormLargeStress4DTestCases()
{
    return {
        {{16, 288, 48, 32}}, // 7077888
        {{16, 576, 1, 30}}, // 276480
        {{16, 2048, 16, 32}}, // 16777216
        {{128, 35, 48, 32}}, // 6881280
    };
}

inline std::vector<BatchnormTestCase> getBatchnormLargeStress5DTestCases()
{
    return {
        {{16, 128, 8, 24, 16}}, // 6291456
        {{32, 128, 8, 16, 16}}, // 8388608
        {{2048, 1024, 2, 1, 1}}, // 4194304
    };
}

} // namespace gpu_batchnorm_ref_test
