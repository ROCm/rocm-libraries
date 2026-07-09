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

} // namespace test_layernorm_common
