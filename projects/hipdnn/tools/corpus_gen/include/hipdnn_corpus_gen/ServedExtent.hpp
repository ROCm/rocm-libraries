// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/OperationMetadata.hpp>
#include <hipdnn_corpus_gen/ProblemSpace.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

/// @file ServedExtent.hpp
/// @brief How far the engine's served region reaches along each numeric parameter, and the
/// points at its edges.
///
/// Spread selection can leave a parameter's largest served value out; a corpus that has to
/// represent what an engine serves reports these edges and can take them. The extent is how far
/// the search reached: a lower bound on the served region, not a proof of its edge.
namespace hipdnn_corpus_gen
{

/// One end of one parameter's served range, and a served point at it.
struct ServedEdge
{
    std::string parameter;
    std::string end; // "low" or "high"
    int64_t value = 0;
    ProblemPoint point;
};

struct ServedExtent
{
    /// parameter -> (smallest, largest) served value, numeric parameters only.
    std::map<std::string, std::pair<int64_t, int64_t>> ranges;

    /// For each parameter in @ref ranges, its low edge then its high edge, in declaration order.
    std::vector<ServedEdge> edges;

    size_t servedPoints = 0;
};

/// @brief The served extent of @p served under @p metadata's numeric parameters.
///
/// Ties go to the point that collates first under `detail::describe`, so a seed reproduces the
/// edges.
inline ServedExtent servedExtent(const OperationMetadata& metadata,
                                 const std::vector<ProblemPoint>& served)
{
    ServedExtent extent;
    extent.servedPoints = served.size();
    std::vector<std::string> keys;
    keys.reserve(served.size());
    for(const auto& point : served)
    {
        keys.push_back(detail::describe(point));
    }
    extent.edges.reserve(metadata.parameters.size() * 2);
    for(const auto& parameter : metadata.parameters)
    {
        if(parameter.type != ParameterType::INT64 && parameter.type != ParameterType::FLOAT64)
        {
            continue;
        }
        std::optional<size_t> low;
        std::optional<size_t> high;
        int64_t lowValue = 0;
        int64_t highValue = 0;
        for(size_t i = 0; i < served.size(); ++i)
        {
            const auto value = detail::integerAt(served[i], parameter.name);
            if(!value.has_value())
            {
                continue;
            }
            if(!low.has_value() || *value < lowValue
               || (*value == lowValue && keys[i] < keys[*low]))
            {
                low = i;
                lowValue = *value;
            }
            if(!high.has_value() || *value > highValue
               || (*value == highValue && keys[i] < keys[*high]))
            {
                high = i;
                highValue = *value;
            }
        }
        if(!low.has_value())
        {
            continue;
        }
        extent.ranges[parameter.name] = {lowValue, highValue};
        extent.edges.push_back({parameter.name, "low", lowValue, served[*low]});
        extent.edges.push_back({parameter.name, "high", highValue, served[*high]});
    }
    return extent;
}

/// @brief Every point the run knows the engine serves: the pooled points and each searched
/// combination's edges, selected or not.
inline std::vector<ProblemPoint> servedPoints(const ProblemCorpus& corpus,
                                              const std::vector<ProblemPoint>& pooled)
{
    std::vector<ProblemPoint> served = pooled;
    for(const auto& combination : corpus.combinations)
    {
        served.insert(served.end(), combination.lowest.begin(), combination.lowest.end());
        served.insert(served.end(), combination.highest.begin(), combination.highest.end());
    }
    return served;
}

} // namespace hipdnn_corpus_gen
