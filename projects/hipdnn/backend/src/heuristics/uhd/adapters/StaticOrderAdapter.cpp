// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "StaticOrderAdapter.hpp"

#include "../FeatureExtractor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

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
    // Pack the ordered fields into one descending score, most-significant field first:
    //   score = -sum_i(RADIX^(n-1-i) * field_i)
    //
    // This only reproduces a true lexicographic order while every term stays exactly
    // representable. Beyond that the low-order fields fall off the end of the mantissa
    // and the ranking silently stops matching priority-then-id, which RFC 0019 §6
    // step 5 requires to be deterministic. Rather than misrank, throw: SelectionEngine
    // marks the candidate invalid, and once no candidate scores it degrades to
    // applyStaticOrdering(), whose direct (priority, id) comparator is exact.
    //
    // TODO(RFC-0019 §6): the real fix is an ordering interface rather than a packed
    // scalar — IUhdAdapter::score() cannot express a lexicographic key. Until then the
    // exactness guard keeps the failure honest.
    if(_orderFieldIndices.empty())
    {
        return 0.0;
    }

    constexpr double kRadix = 1e10;
    // Doubles carry 2^53 of exact integer range; the packed value must stay inside it.
    constexpr double kExactLimit = 9007199254740992.0; // 2^53

    double score = 0.0;
    double weight = std::pow(kRadix, static_cast<double>(_orderFieldIndices.size() - 1));

    size_t position = 0;
    for(const size_t idx : _orderFieldIndices)
    {
        if(idx < features.size())
        {
            const double field = features[idx];
            if(!std::isfinite(field) || field < 0.0)
            {
                throw AdapterOrderingError(
                    "StaticOrderAdapter: order field is negative or non-finite, so the "
                    "packed ordering would not be monotonic");
            }

            // Every field below the most significant one occupies a fixed-width digit.
            // A value at or above the radix carries into the place above it, so a
            // less-significant field can outrank a more-significant one — e.g. with
            // (priority, id), an id of 2e10 outweighs a whole unit of priority.
            if(position > 0 && field >= kRadix)
            {
                throw AdapterOrderingError(
                    "StaticOrderAdapter: order field value " + std::to_string(field) +
                    " is at or above the packing radix, so it would carry into a "
                    "more-significant field and invert the ranking");
            }

            const double term = weight * field;
            if(!std::isfinite(term) || score + term >= kExactLimit)
            {
                throw AdapterOrderingError(
                    "StaticOrderAdapter: packed order value exceeds the range a double "
                    "represents exactly, so the ranking would not match the declared "
                    "field order");
            }

            // Negate so lower field values sort first under a "max" objective.
            score += term;
        }
        weight /= kRadix;
        ++position;
    }

    return -score;
}

} // namespace hipdnn_backend::heuristics::uhd
