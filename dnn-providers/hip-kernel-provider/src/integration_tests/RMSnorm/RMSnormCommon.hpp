// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <ostream>
#include <vector>

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>

namespace hip_kernel_provider::test_rmsnorm_common
{

struct RMSnormTestCase
{
    std::vector<int64_t> dims;
    unsigned int seed;

    RMSnormTestCase(std::vector<int64_t>&& dimsLocal, unsigned int seedLocal)
        : dims(std::move(dimsLocal))
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
        ss << ")";

        return ss;
    }
};

inline std::vector<RMSnormTestCase> getRMSnormTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{2, 3, 4, 4}, seed},
        {{5, 256, 14, 14}, seed},
    };
}

inline std::vector<RMSnormTestCase> getRMSnormFullTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{1, 3, 14, 14}, seed},
        {{1, 256, 1, 1}, seed},
        {{2, 3, 1, 1}, seed},
        {{32, 1, 14, 14}, seed},
        {{32, 3, 1, 14}, seed},
        {{32, 3, 14, 1}, seed},
    };
}

inline std::vector<RMSnormTestCase> getRMSnorm3dTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{2, 3, 3, 1, 1}, seed},
        {{16, 3, 8, 14, 14}, seed},
    };
}

} // namespace hip_kernel_provider::test_rmsnorm_common
