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

inline const char* reductionModeName(hipdnn_frontend::ReductionMode mode)
{
    switch(mode)
    {
    case hipdnn_frontend::ReductionMode::NOT_SET:
        return "NOT_SET";
    case hipdnn_frontend::ReductionMode::ADD:
        return "ADD";
    case hipdnn_frontend::ReductionMode::MUL:
        return "MUL";
    case hipdnn_frontend::ReductionMode::MIN:
        return "MIN";
    case hipdnn_frontend::ReductionMode::MAX:
        return "MAX";
    case hipdnn_frontend::ReductionMode::AMAX:
        return "AMAX";
    case hipdnn_frontend::ReductionMode::AVG:
        return "AVG";
    case hipdnn_frontend::ReductionMode::NORM1:
        return "NORM1";
    case hipdnn_frontend::ReductionMode::NORM2:
        return "NORM2";
    case hipdnn_frontend::ReductionMode::MUL_NO_ZEROS:
        return "MUL_NO_ZEROS";
    default:
        return "UNKNOWN";
    }
}

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
        ss << " mode:" << reductionModeName(tc.mode);
        ss << " seed:" << tc.seed;
        ss << ")";

        return ss;
    }
};

inline std::vector<ReductionTestCase> getReductionTestCases()
{
    using Mode = hipdnn_frontend::ReductionMode;
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
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

        // Shape coverage (ADD mode): various reduction patterns
        // Batched, spatial reduction
        {{4, 8, 4, 4}, {4, 8, 1, 1}, Mode::ADD, seed},
        // Non-square spatial
        {{2, 3, 16, 8}, {2, 3, 1, 1}, Mode::ADD, seed},
        // Batch reduction (reduce dim 0)
        {{4, 8, 4, 4}, {1, 8, 4, 4}, Mode::ADD, seed},
        // Full reduction (all dims)
        {{2, 3, 4, 4}, {1, 1, 1, 1}, Mode::ADD, seed},
        // Single channel
        {{1, 1, 32, 32}, {1, 1, 1, 1}, Mode::ADD, seed},
    };
}

} // namespace test_reduction_common
