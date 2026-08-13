// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <ostream>
#include <vector>

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>

namespace test_pointwise_common
{

struct PointwiseTestCase
{
    std::vector<int64_t> dims;
    unsigned seed;

    friend std::ostream& operator<<(std::ostream& ss, const PointwiseTestCase& tc)
    {
        using namespace hipdnn_data_sdk::utilities;

        ss << "(dims:";
        vecToStream(ss, tc.dims);
        ss << " seed:" << tc.seed << ")";

        return ss;
    }
};

inline std::vector<PointwiseTestCase> getPointwiseTestCases()
{
    const unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{2, 4, 8, 8}, seed},
        {{1, 16, 4, 4}, seed},
        {{4, 8, 16, 16}, seed},
    };
}

} // namespace test_pointwise_common
