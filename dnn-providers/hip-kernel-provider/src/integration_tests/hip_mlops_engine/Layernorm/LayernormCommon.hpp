// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <ostream>
#include <vector>

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>

namespace hip_kernel_provider::layernorm::test::common
{

struct LayernormTestCase
{
    std::vector<int64_t> dims;
    size_t normalizedDim;
    bool weightBias;
    unsigned int seed;

    LayernormTestCase(std::vector<int64_t>&& dimsLocal,
                      size_t normalizedDimLocal,
                      bool weightBiasLocal,
                      unsigned int seedLocal)
        : dims(std::move(dimsLocal))
        , normalizedDim(normalizedDimLocal)
        , weightBias(weightBiasLocal)
        , seed(seedLocal)
    {
        if(dims.size() != 4 && dims.size() != 5)
        {
            throw std::invalid_argument(
                "Dimensions must be either 4D (NCHW, NHWC) or 5D (NCDHW, NDHWC)");
        }
    }

    friend std::ostream& operator<<(std::ostream& ss, const LayernormTestCase& testCase)
    {
        ss << "(dims:";
        hipdnn_data_sdk::utilities::vecToStream(ss, testCase.dims);
        ss << " normalizedDim:" << testCase.normalizedDim;
        ss << " seed:" << testCase.seed;
        ss << ")";

        return ss;
    }
};

inline std::vector<LayernormTestCase> getLayernormFwd4DSmokeTestCases()
{
    unsigned int seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    // clang-format off
    return {
        {{2, 2, 3, 2}, 3, false, seed}, // Minimal test cases
        {{2, 2, 3, 2}, 2, false, seed},
        {{2, 2, 3, 2}, 1, false, seed},
        {{2, 2, 3, 2}, 3, true, seed},
        {{2, 2, 3, 2}, 2, true, seed},
        {{2, 2, 3, 2}, 1, true, seed},
        {{2, 5, 2, 2}, 1, true, seed}, // Edge case: larger C with normalized dim 1
    };
    // clang-format on
};

inline std::vector<LayernormTestCase> getLayernormFwd5DSmokeTestCases()
{
    unsigned int seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    // clang-format off
    return {
        {{2, 2, 3, 2, 2}, 4, false, seed}, // Minimal test cases
        {{2, 2, 3, 2, 2}, 3, false, seed},
        {{2, 2, 3, 2, 2}, 2, false, seed},
        {{2, 2, 3, 2, 2}, 1, false, seed},
        {{2, 2, 3, 2, 2}, 4, true, seed},
        {{2, 2, 3, 2, 2}, 3, true, seed},
        {{2, 2, 3, 2, 2}, 2, true, seed},
        {{2, 2, 3, 2, 2}, 1, true, seed},
        {{2, 5, 2, 2, 2}, 1, true, seed}, // Edge case: larger C with normalized dim 1
    };
    // clang-format on
};

inline std::vector<LayernormTestCase> getLayernormFwd4DFullTestCases()
{
    unsigned int seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    // Imported from MIOpen, most commented out due to excessive testing time
    // clang-format off
    return {
        {{32, 4, 4, 256}, 1, false, seed},
        // {{64, 4, 4, 256}, 1, false, seed},
        {{32, 4, 4, 256}, 1, true, seed},
        // {{64, 4, 4, 256}, 1, true, seed}
    };
    // clang-format on
}

inline std::vector<LayernormTestCase> getLayernormFwd5DFullTestCases()
{
    unsigned int seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    // Imported from MIOpen, most commented out due to excessive testing time
    // clang-format off
    return {
        {{32, 1, 32, 32, 32}, 4, false, seed}, // 32x32x32 based on VoxNet arch
        // {{32, 1, 14, 14, 14}, 4, false, seed},
        // {{32, 32, 14, 14, 14}, 4, false, seed},
        // {{32, 32, 12, 12, 12}, 4, false, seed},
        // {{32, 32, 6, 6, 6}, 4, false, seed},
        // {{256, 1, 32, 32, 32}, 4, false, seed}, // 32x32x32 based on VoxNet arch
        // {{256, 32, 14, 14, 14}, 4, false, seed},
        // {{256, 32, 12, 12, 12}, 4, false, seed},
        // {{256, 32, 6, 6, 6}, 4, false, seed},
        // {{512, 1, 32, 32, 32}, 4, false, seed}, // 32x32x32 based on VoxNet arch
        // {{512, 32, 14, 14, 14}, 4, false, seed},
        // {{512, 32, 12, 12, 12}, 4, false, seed},
        // {{512, 32, 6, 6, 6}, 4, false, seed},
        // {{32, 2, 32, 57, 125}, 4, false, seed}, // Hand-gesture recognition CVPR 2015 paper High Res Net Path
        {{32, 32, 14, 25, 59}, 4, false, seed},
        // {{32, 32, 6, 10, 27}, 4, false, seed},
        // {{32, 32, 4, 6, 11}, 4, false, seed},
        // {{32, 32, 2, 2, 3}, 4, false, seed},
        // {{32, 32, 32, 28, 62}, 4, false, seed}, // Hand-gesture recognition CVPR 2015 paper Low Res Net Path
        // {{32, 32, 14, 12, 29}, 4, false, seed},
        // {{32, 32, 6, 4, 12}, 4, false, seed},
        // {{32, 32, 4, 2, 2}, 4, false, seed},
        // {{16, 32, 6, 50, 50}, 4, false, seed}, // Multi-view 3D convnet
        // {{1, 3, 8, 240, 320}, 4, false, seed}, // 3D convet on video
        // {{1, 3, 16, 240, 320}, 4, false, seed}, // 3D convet on video
        // {{1, 3, 8, 128, 171}, 4, false, seed}, // 3D convet on video
        // {{1, 3, 16, 128, 171}, 4, false, seed}, // 3D convet on video
        // {{1, 3, 8, 112, 112}, 4, false, seed}, // 3D convet on video
        // {{1, 3, 16, 112, 112}, 4, false, seed}, // 3D convet on video
        {{32, 1, 32, 32, 32}, 4, true, seed}, // 32x32x32 based on VoxNet arch
        // {{32, 1, 14, 14, 14}, 4, true, seed},
        // {{32, 32, 14, 14, 14}, 4, true, seed},
        // {{32, 32, 12, 12, 12}, 4, true, seed},
        // {{32, 32, 6, 6, 6}, 4, true, seed},
        // {{256, 1, 32, 32, 32}, 4, true, seed}, // 32x32x32 based on VoxNet arch
        // {{256, 32, 14, 14, 14}, 4, true, seed},
        // {{256, 32, 12, 12, 12}, 4, true, seed},
        // {{256, 32, 6, 6, 6}, 4, true, seed},
        // {{512, 1, 32, 32, 32}, 4, true, seed}, // 32x32x32 based on VoxNet arch
        // {{512, 32, 14, 14, 14}, 4, true, seed},
        // {{512, 32, 12, 12, 12}, 4, true, seed},
        // {{512, 32, 6, 6, 6}, 4, true, seed},
        // {{32, 2, 32, 57, 125}, 4, true, seed}, // Hand-gesture recognition CVPR 2015 paper High Res Net Path
        {{32, 32, 14, 25, 59}, 4, true, seed},
        // {{32, 32, 6, 10, 27}, 4, true, seed},
        // {{32, 32, 4, 6, 11}, 4, true, seed},
        // {{32, 32, 2, 2, 3}, 4, true, seed},
        // {{32, 32, 32, 28, 62}, 4, true, seed}, // Hand-gesture recognition CVPR 2015 paper Low Res Net Path
        // {{32, 32, 14, 12, 29}, 4, true, seed},
        // {{32, 32, 6, 4, 12}, 4, true, seed},
        // {{32, 32, 4, 2, 2}, 4, true, seed},
        // {{16, 32, 6, 50, 50}, 4, true, seed}, // Multi-view 3D convnet
        // {{1, 3, 8, 240, 320}, 4, true, seed}, // 3D convet on video
        // {{1, 3, 16, 240, 320}, 4, true, seed}, // 3D convet on video
        // {{1, 3, 8, 128, 171}, 4, true, seed}, // 3D convet on video
        // {{1, 3, 16, 128, 171}, 4, true, seed}, // 3D convet on video
        // {{1, 3, 8, 112, 112}, 4, true, seed}, // 3D convet on video
        // {{1, 3, 16, 112, 112}, 4, true, seed} // 3D convet on video
    };
    // clang-format on
}

} // namespace hip_kernel_provider::layernorm::test::common
