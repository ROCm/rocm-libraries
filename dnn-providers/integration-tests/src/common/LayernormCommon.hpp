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
    bool optionalTensors;
    unsigned int seed;

    LayernormTestCase(std::vector<int64_t>&& dimsLocal,
                      size_t normalizedDimLocal,
                      bool optionalTensorsLocal,
                      unsigned int seedLocal)
        : dims(std::move(dimsLocal))
        , normalizedDim(normalizedDimLocal)
        , optionalTensors(optionalTensorsLocal)
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
        ss << " optionalTensors:" << tc.optionalTensors;
        ss << " seed:" << tc.seed;
        ss << ")";

        return ss;
    }
};

// ============================================================================
// Tiered shape catalogs.
//
// The tiers are disjoint: every (dims, normalizedDim, optionalTensors) triple
// appears in exactly one function per dimensionality. No tier is a superset of
// another, so a tier costs exactly its own entries and CTest's cumulative
// labels (quick -> standard -> comprehensive -> full) do the widening.
//
// Membership is by tensor element count. Runtime is dominated by the harness's
// host-side verification at roughly 450 ns per element regardless of dtype, and
// every entry below is multiplied by 2 layouts x 7 dtype fixtures:
//
//   Quick          < 1K elements     normalization-axis and boundary sweep
//   Standard       <= 400K elements  small production shapes
//   Comprehensive  <= 5M elements    mid-size production shapes
//   Full           > 5M elements     heaviest production and large-batch shapes
//
// Production shapes are imported from the MIOpen layernorm suite. 4D tops out
// at 262K elements, so it has no Comprehensive or Full tier.
// ============================================================================

// 4D Quick: normalization boundary swept across every axis on a minimal tensor.
// Sub-millisecond; this is the pre-commit signal that each axis is wired correctly.
inline std::vector<LayernormTestCase> getLayernorm4DQuickTestCases()
{
    const unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{2, 2, 3, 2}, 1, false, seed},
        {{2, 2, 3, 2}, 1, true, seed},
        {{2, 2, 3, 2}, 2, false, seed},
        {{2, 2, 3, 2}, 2, true, seed},
        {{2, 2, 3, 2}, 3, false, seed},
        {{2, 2, 3, 2}, 3, true, seed},
        {{2, 5, 2, 2}, 1, true, seed},
    };
}

// 4D Standard: the only production-scale 4D shapes in the suite (262K elements at
// most), so 4D is fully covered by the PR gate and has no Comprehensive or Full tier.
inline std::vector<LayernormTestCase> getLayernorm4DStandardTestCases()
{
    const unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{32, 4, 4, 256}, 1, false, seed},
        {{32, 4, 4, 256}, 1, true, seed},
        {{64, 4, 4, 256}, 1, false, seed},
        {{64, 4, 4, 256}, 1, true, seed},
    };
}

// 5D Quick: same normalization-axis sweep as the 4D Quick set, one dimension up.
inline std::vector<LayernormTestCase> getLayernorm5DQuickTestCases()
{
    const unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{2, 2, 3, 2, 2}, 1, false, seed},
        {{2, 2, 3, 2, 2}, 1, true, seed},
        {{2, 2, 3, 2, 2}, 2, false, seed},
        {{2, 2, 3, 2, 2}, 2, true, seed},
        {{2, 2, 3, 2, 2}, 3, false, seed},
        {{2, 2, 3, 2, 2}, 3, true, seed},
        {{2, 2, 3, 2, 2}, 4, false, seed},
        {{2, 2, 3, 2, 2}, 4, true, seed},
        {{2, 5, 2, 2, 2}, 1, true, seed},
    };
}

// 5D Standard: small production shapes, all under 400K elements.
inline std::vector<LayernormTestCase> getLayernorm5DStandardTestCases()
{
    const unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{32, 32, 2, 2, 3}, 4, false, seed},
        {{32, 32, 2, 2, 3}, 4, true, seed},
        {{32, 32, 4, 2, 2}, 4, false, seed},
        {{32, 32, 4, 2, 2}, 4, true, seed},
        {{32, 1, 14, 14, 14}, 4, false, seed}, // VoxNet-style volumetric
        {{32, 1, 14, 14, 14}, 4, true, seed},
        {{32, 32, 6, 6, 6}, 4, false, seed},
        {{32, 32, 6, 6, 6}, 4, true, seed},
        {{32, 32, 4, 6, 11}, 4, false, seed},
        {{32, 32, 4, 6, 11}, 4, true, seed},
        {{32, 32, 6, 4, 12}, 4, false, seed},
        {{32, 32, 6, 4, 12}, 4, true, seed},
        {{1, 3, 8, 112, 112}, 4, false, seed}, // 3D convnet on video
        {{1, 3, 8, 112, 112}, 4, true, seed},
    };
}

// 5D Comprehensive: mid-size production shapes, 400K to 5M elements.
inline std::vector<LayernormTestCase> getLayernorm5DComprehensiveTestCases()
{
    const unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{1, 3, 8, 128, 171}, 4, false, seed}, // 3D convnet on video
        {{1, 3, 8, 128, 171}, 4, true, seed},
        {{1, 3, 16, 112, 112}, 4, false, seed}, // 3D convnet on video
        {{1, 3, 16, 112, 112}, 4, true, seed},
        {{32, 1, 32, 32, 32}, 4, false, seed}, // VoxNet-style 32x32x32 volumetric
        {{32, 1, 32, 32, 32}, 4, true, seed},
        {{1, 3, 16, 128, 171}, 4, false, seed}, // 3D convnet on video
        {{1, 3, 16, 128, 171}, 4, true, seed},
        {{32, 32, 6, 10, 27}, 4, false, seed},
        {{32, 32, 6, 10, 27}, 4, true, seed},
        {{32, 32, 12, 12, 12}, 4, false, seed},
        {{32, 32, 12, 12, 12}, 4, true, seed},
        {{1, 3, 8, 240, 320}, 4, false, seed}, // 3D convnet on video
        {{1, 3, 8, 240, 320}, 4, true, seed},
        {{32, 32, 14, 14, 14}, 4, false, seed},
        {{32, 32, 14, 14, 14}, 4, true, seed},
        {{1, 3, 16, 240, 320}, 4, false, seed}, // 3D convnet on video
        {{1, 3, 16, 240, 320}, 4, true, seed},
        {{32, 32, 14, 12, 29}, 4, false, seed},
        {{32, 32, 14, 12, 29}, 4, true, seed},
    };
}

// 5D Full: the heaviest shapes, above 5M elements, including the batch-256/512
// volumetric set. Roughly 40 minutes across both directions - weekly tier only.
inline std::vector<LayernormTestCase> getLayernorm5DFullTestCases()
{
    const unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{16, 32, 6, 50, 50}, 4, false, seed}, // Multi-view 3D convnet
        {{16, 32, 6, 50, 50}, 4, true, seed},
        {{256, 1, 32, 32, 32}, 4, false, seed}, // batch-256 volumetric
        {{256, 1, 32, 32, 32}, 4, true, seed},
        {{32, 2, 32, 57, 125},
         4,
         false,
         seed}, // Hand-gesture recognition (CVPR 2015) high-res path
        {{32, 2, 32, 57, 125}, 4, true, seed},
        {{512, 1, 32, 32, 32}, 4, false, seed}, // batch-512 volumetric
        {{512, 1, 32, 32, 32}, 4, true, seed},
        {{32, 32, 14, 25, 59}, 4, false, seed},
        {{32, 32, 14, 25, 59}, 4, true, seed},
        {{256, 32, 14, 14, 14}, 4, false, seed}, // batch-256 volumetric
        {{256, 32, 14, 14, 14}, 4, true, seed},
        {{512, 32, 14, 14, 14}, 4, false, seed}, // batch-512 volumetric
        {{512, 32, 14, 14, 14}, 4, true, seed},
        {{32, 32, 32, 28, 62}, 4, false, seed}, // Hand-gesture recognition (CVPR 2015) low-res path
        {{32, 32, 32, 28, 62}, 4, true, seed},
    };
}

} // namespace test_layernorm_common
