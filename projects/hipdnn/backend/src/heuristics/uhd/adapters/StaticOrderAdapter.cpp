// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "StaticOrderAdapter.hpp"

#include "../FeatureExtractor.hpp"

#include <algorithm>
#include <cmath>

namespace hipdnn_backend::heuristics::uhd
{

const std::string StaticOrderAdapter::EMPTY_HASH;

StaticOrderAdapter::StaticOrderAdapter(std::vector<size_t> orderFieldIndices, size_t numFeatures)
    : _orderFieldIndices(std::move(orderFieldIndices)), _numFeatures(numFeatures)
{
}

std::unique_ptr<StaticOrderAdapter>
    StaticOrderAdapter::create(const std::vector<std::string>& orderFields,
                               const std::vector<std::string>& signature)
{
    // Resolve each signature entry to its reference name once, so the bare
    // (`$kernel.priority`) and pre-quoted (`"\"$kernel.priority\""`) spellings both
    // match. Non-reference entries (derived expressions) resolve to empty and are
    // never matched by an order field.
    std::vector<std::string> refNames;
    refNames.reserve(signature.size());
    for(const auto& entry : signature)
    {
        nlohmann::json parsed;
        try
        {
            parsed = FeatureExtractor::parseSignatureEntry(entry);
        }
        catch(const JsonLogicError&)
        {
            refNames.emplace_back();
            continue;
        }
        refNames.push_back(parsed.is_string() ? parsed.get<std::string>() : std::string{});
    }

    std::vector<size_t> indices;
    indices.reserve(orderFields.size());

    for(const auto& field : orderFields)
    {
        // Order fields may be given bare (`priority`) or namespaced (`$kernel.priority`).
        bool found = false;
        for(size_t i = 0; i < refNames.size(); ++i)
        {
            if(!refNames[i].empty() &&
               (refNames[i] == field || refNames[i] == "$kernel." + field))
            {
                indices.push_back(i);
                found = true;
                break;
            }
        }
        if(!found)
        {
            return nullptr;
        }
    }

    return std::make_unique<StaticOrderAdapter>(std::move(indices), signature.size());
}

double StaticOrderAdapter::score(const std::vector<double>& features) const
{
    // Compute a composite score where earlier fields have exponentially higher weight.
    // Lower field values yield higher scores (for "max" objective ranking).
    //
    // Score = sum_i(weight_i * -field_i)
    // where weight_i = 1e10^(numFields - i - 1)
    //
    // This ensures field[0] dominates, then field[1], etc.

    if(_orderFieldIndices.empty())
    {
        return 0.0;
    }

    double score = 0.0;
    double weight = std::pow(1e10, static_cast<double>(_orderFieldIndices.size() - 1));

    for(const size_t idx : _orderFieldIndices)
    {
        if(idx < features.size())
        {
            // Negate so lower values get higher scores
            score += weight * (-features[idx]);
        }
        weight /= 1e10;
    }

    return score;
}

} // namespace hipdnn_backend::heuristics::uhd
