// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace test_rmsnorm_common
{

struct RMSNormTestCase
{
    std::vector<int64_t> xDims;
    std::vector<int64_t> scaleDims;
    std::optional<std::vector<int64_t>> biasDims;
    bool isTraining;
    float epsilon;
    unsigned seed;

    RMSNormTestCase(std::vector<int64_t>&& xDimsLocal,
                    std::vector<int64_t>&& scaleDimsLocal,
                    std::optional<std::vector<int64_t>>&& biasDimsLocal,
                    bool isTrainingLocal,
                    float epsilonLocal,
                    unsigned seedLocal)
        : xDims(std::move(xDimsLocal))
        , scaleDims(std::move(scaleDimsLocal))
        , biasDims(std::move(biasDimsLocal))
        , isTraining(isTrainingLocal)
        , epsilon(epsilonLocal)
        , seed(seedLocal)
    {
        if(xDims.size() != scaleDims.size())
        {
            throw std::invalid_argument("xDims and scaleDims must have the same rank.");
        }

        if(xDims.size() < 2)
        {
            throw std::invalid_argument("xDims must have at least 2 dimensions.");
        }

        if(scaleDims[0] != 1)
        {
            throw std::invalid_argument(
                "scaleDims[0] must be 1 (scale is broadcast across batch).");
        }

        if(biasDims.has_value() && *biasDims != scaleDims)
        {
            throw std::invalid_argument("biasDims must equal scaleDims when set.");
        }
    }

    friend std::ostream& operator<<(std::ostream& ss, const RMSNormTestCase& tc)
    {
        using namespace hipdnn_data_sdk::utilities;

        ss << "(x:";
        vecToStream(ss, tc.xDims);
        ss << " scale:";
        vecToStream(ss, tc.scaleDims);
        if(tc.biasDims.has_value())
        {
            ss << " bias:";
            vecToStream(ss, *tc.biasDims);
        }
        ss << " phase:" << (tc.isTraining ? "TRAINING" : "INFERENCE");
        ss << " eps:" << tc.epsilon;
        ss << " seed:" << tc.seed;
        ss << ")";

        return ss;
    }
};

inline std::vector<RMSNormTestCase> getRMSNormTestCases()
{
    const float eps = 1e-5f;
    const unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    const std::vector<std::tuple<std::vector<int64_t>, std::vector<int64_t>>> shapes = {
        {{2, 16, 8, 8}, {1, 16, 8, 8}}, // Normalized shape [C, H, W]
        {{2, 16, 8, 8}, {1, 1, 8, 8}}, // Normalized shape [H, W]
        {{2, 16, 8, 8}, {1, 1, 1, 8}}, // Normalized shape [W]
        {{2, 1, 1, 1}, {1, 1, 1, 1}}, // degenerate all-1
        {{4096, 128, 1, 1},
         {1, 128, 1, 1}}, // [batch * sequence_length, hidden_dim, 1, 1], normalize over hidden_dim
        {{32, 3, 1, 14}, {1, 3, 1, 14}}, // degenerate H
        {{32, 3, 14, 1}, {1, 3, 14, 1}}, // degenerate W
    };

    std::vector<RMSNormTestCase> cases;
    cases.reserve(shapes.size() * 4);
    for(const auto& [xDims, scaleDims] : shapes)
    {
        for(const bool isTraining : {false, true})
        {
            for(const bool withBias : {false, true})
            {
                std::optional<std::vector<int64_t>> biasDims
                    = withBias ? std::optional<std::vector<int64_t>>(scaleDims) : std::nullopt;
                cases.emplace_back(std::vector<int64_t>(xDims),
                                   std::vector<int64_t>(scaleDims),
                                   std::move(biasDims),
                                   isTraining,
                                   eps,
                                   seed);
            }
        }
    }
    return cases;
}

} // namespace test_rmsnorm_common
