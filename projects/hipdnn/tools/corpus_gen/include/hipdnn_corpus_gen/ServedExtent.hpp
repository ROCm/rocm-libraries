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
/// A corpus that is to represent what an engine serves -- an evaluation set, scoring a model on
/// everything it may be asked to choose for -- has to reach the edges of that region, and say
/// how far they are. Spread selection does not guarantee the first: it places points over the
/// interior and can leave the largest served `seqlen_k` out in favour of a better-spread one.
/// Nothing reported the second. This is both: per numeric parameter, the smallest and largest
/// value over every point the engine is known to serve, and a served point carrying each.
///
/// The extent is how far the search reached, which is a lower bound on the served region and
/// never a proof of its edge: nothing declares a parameter's range, and a walk can stop short.
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
/// Ties go to the point that collates first under `detail::describe`, so the same served set
/// yields the same edges -- a corpus regenerated from one seed takes the same points.
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

/// @brief Every point the run knows the engine serves: the pooled points, and each searched
/// combination's edges -- which come from every point its walk was answered yes for, selected
/// or not.
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
