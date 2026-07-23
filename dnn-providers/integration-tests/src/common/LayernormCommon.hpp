// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace test_layernorm_common
{

struct LayernormTestCase
{
    std::vector<int64_t> dims;
    size_t normalizedDim;
    bool isTraining;
    unsigned int seed;

    LayernormTestCase(std::vector<int64_t>&& dimsLocal,
                      size_t normalizedDimLocal,
                      bool isTrainingLocal,
                      unsigned int seedLocal)
        : dims(std::move(dimsLocal))
        , normalizedDim(normalizedDimLocal)
        , isTraining(isTrainingLocal)
        , seed(seedLocal)
    {
        if(dims.size() != 4 && dims.size() != 5)
        {
            throw std::invalid_argument(
                "LayernormTestCase requires dims to be 4D (N, C, H, W) or 5D (N, C, D, H, W)");
        }
        if(normalizedDim == 0 || normalizedDim >= dims.size())
        {
            throw std::invalid_argument("normalizedDim must be in [1, dims.size() - 1]");
        }
    }

    friend std::ostream& operator<<(std::ostream& ss, const LayernormTestCase& tc)
    {
        using namespace hipdnn_data_sdk::utilities;

        ss << "(dims:";
        vecToStream(ss, tc.dims);
        ss << " normalizedDim:" << tc.normalizedDim;
        ss << " phase:" << (tc.isTraining ? "TRAINING" : "INFERENCE");
        ss << " seed:" << tc.seed;
        ss << ")";

        return ss;
    }
};

// 4D (N, C, H, W) shapes: normalization boundary swept across every axis on a
// small tensor, plus a couple of larger, closer-to-production shapes.
inline std::vector<LayernormTestCase> getLayernormFwd4DTestCases()
{
    const unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{2, 2, 3, 2}, 3, false, seed},
        {{2, 2, 3, 2}, 2, false, seed},
        {{2, 2, 3, 2}, 1, false, seed},
        {{2, 2, 3, 2}, 3, true, seed},
        {{2, 2, 3, 2}, 2, true, seed},
        {{2, 2, 3, 2}, 1, true, seed},
        {{2, 5, 2, 2}, 1, true, seed}, // larger C, normalized over C
        {{32, 4, 4, 256}, 1, false, seed},
        {{32, 4, 4, 256}, 1, true, seed},
    };
}

// 5D (N, C, D, H, W) shapes: same axis sweep as the 4D cases, plus a couple of
// volumetric (VoxNet-style) shapes.
inline std::vector<LayernormTestCase> getLayernormFwd5DTestCases()
{
    const unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{2, 2, 3, 2, 2}, 4, false, seed},
        {{2, 2, 3, 2, 2}, 3, false, seed},
        {{2, 2, 3, 2, 2}, 2, false, seed},
        {{2, 2, 3, 2, 2}, 1, false, seed},
        {{2, 2, 3, 2, 2}, 4, true, seed},
        {{2, 2, 3, 2, 2}, 3, true, seed},
        {{2, 2, 3, 2, 2}, 2, true, seed},
        {{2, 2, 3, 2, 2}, 1, true, seed},
        {{2, 5, 2, 2, 2}, 1, true, seed}, // larger C, normalized over C
        {{32, 1, 32, 32, 32}, 4, false, seed}, // 32x32x32 volumetric shape
        {{32, 32, 14, 25, 59}, 4, false, seed},
        {{32, 1, 32, 32, 32}, 4, true, seed},
        {{32, 32, 14, 25, 59}, 4, true, seed},
    };
}

// Larger, closer-to-production shapes reserved for the Full tier (imported from the MIOpen
// layernorm suite). The heaviest batch-256/512 volumetric shapes live in a separate
// getLayernormFwd5DLargeBatchTestCases() set below so they can be gated independently.
inline std::vector<LayernormTestCase> getLayernormFwd4DFullTestCases()
{
    const unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{64, 4, 4, 256}, 1, false, seed},
        {{64, 4, 4, 256}, 1, true, seed},
    };
}

inline std::vector<LayernormTestCase> getLayernormFwd5DFullTestCases()
{
    const unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{32, 1, 14, 14, 14}, 4, false, seed}, // VoxNet-style volumetric shapes
        {{32, 32, 14, 14, 14}, 4, false, seed},
        {{32, 32, 12, 12, 12}, 4, false, seed},
        {{32, 32, 6, 6, 6}, 4, false, seed},
        {{32, 2, 32, 57, 125},
         4,
         false,
         seed}, // Hand-gesture recognition (CVPR 2015) high-res path
        {{32, 32, 6, 10, 27}, 4, false, seed},
        {{32, 32, 4, 6, 11}, 4, false, seed},
        {{32, 32, 2, 2, 3}, 4, false, seed},
        {{32, 32, 32, 28, 62}, 4, false, seed}, // Hand-gesture recognition (CVPR 2015) low-res path
        {{32, 32, 14, 12, 29}, 4, false, seed},
        {{32, 32, 6, 4, 12}, 4, false, seed},
        {{32, 32, 4, 2, 2}, 4, false, seed},
        {{16, 32, 6, 50, 50}, 4, false, seed}, // Multi-view 3D convnet
        {{1, 3, 8, 240, 320}, 4, false, seed}, // 3D convnet on video
        {{1, 3, 16, 240, 320}, 4, false, seed},
        {{1, 3, 8, 128, 171}, 4, false, seed},
        {{1, 3, 16, 128, 171}, 4, false, seed},
        {{1, 3, 8, 112, 112}, 4, false, seed},
        {{1, 3, 16, 112, 112}, 4, false, seed},
        {{32, 1, 14, 14, 14}, 4, true, seed},
        {{32, 32, 14, 14, 14}, 4, true, seed},
        {{32, 32, 12, 12, 12}, 4, true, seed},
        {{32, 32, 6, 6, 6}, 4, true, seed},
        {{32, 2, 32, 57, 125}, 4, true, seed},
        {{32, 32, 6, 10, 27}, 4, true, seed},
        {{32, 32, 4, 6, 11}, 4, true, seed},
        {{32, 32, 2, 2, 3}, 4, true, seed},
        {{32, 32, 32, 28, 62}, 4, true, seed},
        {{32, 32, 14, 12, 29}, 4, true, seed},
        {{32, 32, 6, 4, 12}, 4, true, seed},
        {{32, 32, 4, 2, 2}, 4, true, seed},
        {{16, 32, 6, 50, 50}, 4, true, seed},
        {{1, 3, 8, 240, 320}, 4, true, seed},
        {{1, 3, 16, 240, 320}, 4, true, seed},
        {{1, 3, 8, 128, 171}, 4, true, seed},
        {{1, 3, 16, 128, 171}, 4, true, seed},
        {{1, 3, 8, 112, 112}, 4, true, seed},
        {{1, 3, 16, 112, 112}, 4, true, seed},
    };
}

// Batch-256/512 volumetric shapes. Measured at ~17-29 s per case and roughly doubling the
// 5D layernorm full-tier runtime, so they are instantiated under a dedicated
// "Full5dLargeBatch" prefix and currently skipped via each engine's test-config TOML. They
// add batch scale over the batch-32 {*,*,14,14,14}/{*,1,32,32,32} shapes already in the Full
// set (no distinct code path). Drop the TOML skip once per-test tier filtering is fully wired.
inline std::vector<LayernormTestCase> getLayernormFwd5DLargeBatchTestCases()
{
    const unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{256, 1, 32, 32, 32}, 4, false, seed},
        {{256, 32, 14, 14, 14}, 4, false, seed},
        {{512, 1, 32, 32, 32}, 4, false, seed},
        {{512, 32, 14, 14, 14}, 4, false, seed},
        {{256, 1, 32, 32, 32}, 4, true, seed},
        {{256, 32, 14, 14, 14}, 4, true, seed},
        {{512, 1, 32, 32, 32}, 4, true, seed},
        {{512, 32, 14, 14, 14}, 4, true, seed},
    };
}

} // namespace test_layernorm_common
