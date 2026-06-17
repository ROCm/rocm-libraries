// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <vector>

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>

namespace hip_kernel_provider::rmsnorm::test::common
{

struct RMSnormTestCase
{
    std::vector<int64_t> ioDims;
    std::vector<int64_t> scaleDims;

    bool isTraining; // True for training or false for inference
    unsigned int seed;

    RMSnormTestCase(std::vector<int64_t>&& ioLocal,
                    std::vector<int64_t>&& scaleLocal,
                    bool isTrainingLocal,
                    unsigned int seedLocal)
        : ioDims(std::move(ioLocal))
        , scaleDims(std::move(scaleLocal))
        , isTraining(isTrainingLocal)
        , seed(seedLocal)
    {
        if(ioDims.size() != 4 && ioDims.size() != 5)
        {
            throw std::invalid_argument(
                "dims must be either 4D (N, C, H, W) or 5D (N, C, D, H, W)");
        }
        if(ioDims.size() != scaleDims.size())
        {
            throw std::invalid_argument("io and scale must have the same number of dimensions");
        }
    }

    friend std::ostream& operator<<(std::ostream& ss, const RMSnormTestCase& tc)
    {
        ss << "(io dims:";
        hipdnn_data_sdk::utilities::vecToStream(ss, tc.ioDims);
        ss << " scale dims:";
        hipdnn_data_sdk::utilities::vecToStream(ss, tc.scaleDims);
        ss << " seed:" << tc.seed;
        ss << " isTraining: " << tc.isTraining;
        ss << ")";

        return ss;
    }
};

inline std::vector<RMSnormTestCase> getRMSnormTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{2, 3, 4, 4}, {1, 3, 4, 4}, false, seed},
        {{5, 256, 14, 14}, {1, 256, 14, 14}, false, seed},
        {{2, 3, 4, 4}, {1, 3, 4, 4}, true, seed},
        {{2, 3, 4, 4}, {1, 1, 4, 4}, true, seed},
        {{5, 256, 14, 14}, {1, 256, 14, 14}, true, seed},
    };
}

inline std::vector<RMSnormTestCase> getRMSnormFullTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{1, 3, 14, 14}, {1, 3, 14, 14}, false, seed},
        {{1, 3, 14, 14}, {1, 1, 14, 14}, false, seed},
        {{1, 3, 14, 14}, {1, 1, 1, 14}, false, seed},
        {{1, 256, 1, 1}, {1, 256, 1, 1}, false, seed},
        {{2, 3, 1, 1}, {1, 3, 1, 1}, false, seed},
        {{32, 1, 14, 14}, {1, 1, 14, 14}, false, seed},
        {{32, 3, 1, 14}, {1, 3, 1, 14}, false, seed},
        {{32, 3, 14, 1}, {1, 3, 14, 1}, false, seed},
        {{32, 3, 14, 1}, {1, 1, 14, 1}, false, seed},
        {{1, 3, 14, 14}, {1, 3, 14, 14}, true, seed},
        {{1, 3, 14, 14}, {1, 1, 14, 14}, true, seed},
        {{1, 3, 14, 14}, {1, 1, 1, 14}, true, seed},
        {{1, 256, 1, 1}, {1, 256, 1, 1}, true, seed},
        {{2, 3, 1, 1}, {1, 3, 1, 1}, true, seed},
        {{32, 1, 14, 14}, {1, 1, 14, 14}, true, seed},
        {{32, 3, 1, 14}, {1, 3, 1, 14}, true, seed},
        {{32, 3, 14, 1}, {1, 3, 14, 1}, true, seed},
        {{32, 3, 14, 1}, {1, 1, 14, 1}, true, seed},

    };
}

inline std::vector<RMSnormTestCase> getRMSnorm3dTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{2, 3, 3, 1, 1}, {1, 3, 3, 1, 1}, false, seed},
        {{16, 3, 8, 14, 14}, {1, 3, 8, 14, 14}, false, seed},
        {{2, 3, 4, 2, 2}, {1, 1, 4, 2, 2}, false, seed},
        {{2, 3, 4, 2, 2}, {1, 1, 1, 2, 2}, false, seed},
        {{2, 3, 4, 2, 2}, {1, 1, 1, 1, 2}, false, seed},
        {{2, 3, 3, 1, 1}, {1, 3, 3, 1, 1}, true, seed},
        {{16, 3, 8, 14, 14}, {1, 3, 8, 14, 14}, true, seed},
        {{2, 3, 4, 2, 2}, {1, 1, 4, 2, 2}, true, seed},
        {{2, 3, 4, 2, 2}, {1, 1, 1, 2, 2}, true, seed},
        {{2, 3, 4, 2, 2}, {1, 1, 1, 1, 2}, true, seed},
    };
}

} // namespace hip_kernel_provider::rmsnorm::test::common
