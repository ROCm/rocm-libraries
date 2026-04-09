// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace test_reduction_common
{

struct ReductionTestCase
{
    std::vector<int64_t> xDims;
    std::vector<int64_t> yDims;
    hipdnn_frontend::ReductionMode mode;
    unsigned seed;

    ReductionTestCase(std::vector<int64_t>&& xDimsLocal,
                      std::vector<int64_t>&& yDimsLocal,
                      hipdnn_frontend::ReductionMode modeLocal,
                      unsigned seedLocal)
        : xDims(std::move(xDimsLocal))
        , yDims(std::move(yDimsLocal))
        , mode(modeLocal)
        , seed(seedLocal)
    {
        if(xDims.size() != yDims.size())
        {
            throw std::invalid_argument("xDims and yDims must have the same rank.");
        }

        if(xDims.size() < 2)
        {
            throw std::invalid_argument("xDims must have at least 2 dimensions.");
        }

        bool hasReduction = false;
        for(size_t i = 0; i < xDims.size(); ++i)
        {
            if(yDims[i] < xDims[i])
            {
                if(yDims[i] != 1)
                {
                    throw std::invalid_argument("Reduced dimension " + std::to_string(i)
                                                + " must be 1, got " + std::to_string(yDims[i]));
                }
                hasReduction = true;
            }
            else if(yDims[i] != xDims[i])
            {
                throw std::invalid_argument("Non-reduced dimension " + std::to_string(i)
                                            + " must match input, got Y=" + std::to_string(yDims[i])
                                            + " X=" + std::to_string(xDims[i]));
            }
        }

        if(!hasReduction)
        {
            throw std::invalid_argument("At least one dimension must be reduced.");
        }
    }

    friend std::ostream& operator<<(std::ostream& ss, const ReductionTestCase& tc)
    {
        using namespace hipdnn_data_sdk::utilities;

        ss << "(x:";
        vecToStream(ss, tc.xDims);
        ss << " y:";
        vecToStream(ss, tc.yDims);
        ss << " mode:" << tc.mode;
        ss << " seed:" << tc.seed;
        ss << ")";

        return ss;
    }
};

inline std::vector<ReductionTestCase> getReductionTestCases()
{
    using Mode = hipdnn_frontend::ReductionMode;
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    std::vector<ReductionTestCase> cases = {
        // Mode coverage: each of the 9 modes with spatial reduction
        {{1, 16, 8, 8}, {1, 16, 1, 1}, Mode::ADD, seed},
        {{1, 16, 8, 8}, {1, 16, 1, 1}, Mode::MUL, seed},
        {{1, 16, 8, 8}, {1, 16, 1, 1}, Mode::MIN, seed},
        {{1, 16, 8, 8}, {1, 16, 1, 1}, Mode::MAX, seed},
        {{1, 16, 8, 8}, {1, 16, 1, 1}, Mode::AMAX, seed},
        {{1, 16, 8, 8}, {1, 16, 1, 1}, Mode::AVG, seed},
        {{1, 16, 8, 8}, {1, 16, 1, 1}, Mode::NORM1, seed},
        {{1, 16, 8, 8}, {1, 16, 1, 1}, Mode::NORM2, seed},
        {{1, 16, 8, 8}, {1, 16, 1, 1}, Mode::MUL_NO_ZEROS, seed},

    };

    // Shape coverage × {ADD, MAX}: various reduction patterns
    const std::vector<Mode> shapeModes = {Mode::ADD, Mode::MAX};
    const std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> shapes = {
        {{4, 8, 4, 4}, {4, 8, 1, 1}}, // Batched, spatial reduction
        {{2, 3, 16, 8}, {2, 3, 1, 1}}, // Non-square spatial
        {{4, 8, 4, 4}, {1, 8, 4, 4}}, // Batch reduction (reduce dim 0)
        {{2, 3, 4, 4}, {1, 1, 1, 1}}, // Full reduction (all dims)
        {{1, 1, 32, 32}, {1, 1, 1, 1}}, // Single channel
    };
    for(const auto& [x, y] : shapes)
    {
        for(const auto& mode : shapeModes)
        {
            cases.emplace_back(std::vector<int64_t>(x), std::vector<int64_t>(y), mode, seed);
        }
    }

    return cases;
}

} // namespace test_reduction_common
