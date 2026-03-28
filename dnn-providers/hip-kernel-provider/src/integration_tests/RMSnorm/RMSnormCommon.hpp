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
    std::vector<int64_t> dims;
    bool isTraining; // True for training or false for inference
    unsigned int seed;

    RMSnormTestCase(std::vector<int64_t>&& dimsLocal, bool isTrainingLocal, unsigned int seedLocal)
        : dims(std::move(dimsLocal))
        , isTraining(isTrainingLocal)
        , seed(seedLocal)
    {
        if(dims.size() != 4 && dims.size() != 5)
        {
            throw std::invalid_argument(
                "dims must be either 4D (N, C, H, W) or 5D (N, C, D, H, W)");
        }
    }

    friend std::ostream& operator<<(std::ostream& ss, const RMSnormTestCase& tc)
    {
        ss << "(dims:";
        hipdnn_data_sdk::utilities::vecToStream(ss, tc.dims);
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
        {{2, 3, 4, 4}, false, seed},
        {{5, 256, 14, 14}, false, seed},
        {{2, 3, 4, 4}, true, seed},
        {{5, 256, 14, 14}, true, seed},
    };
}

inline std::vector<RMSnormTestCase> getRMSnormFullTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{1, 3, 14, 14}, false, seed},
        {{1, 256, 1, 1}, false, seed},
        {{2, 3, 1, 1}, false, seed},
        {{32, 1, 14, 14}, false, seed},
        {{32, 3, 1, 14}, false, seed},
        {{32, 3, 14, 1}, false, seed},
        {{1, 3, 14, 14}, true, seed},
        {{1, 256, 1, 1}, true, seed},
        {{2, 3, 1, 1}, true, seed},
        {{32, 1, 14, 14}, true, seed},
        {{32, 3, 1, 14}, true, seed},
        {{32, 3, 14, 1}, true, seed},
    };
}

inline std::vector<RMSnormTestCase> getRMSnorm3dTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{2, 3, 3, 1, 1}, false, seed},
        {{16, 3, 8, 14, 14}, false, seed},
        {{2, 3, 3, 1, 1}, true, seed},
        {{16, 3, 8, 14, 14}, true, seed},
    };
}

} // namespace hip_kernel_provider::rmsnorm::test::common
