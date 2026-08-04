// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "IUhdAdapter.hpp"

#include <string>
#include <vector>

namespace hipdnn_backend::heuristics::uhd
{

/// @brief Static order adapter - ranks by field values in declared order.
///
/// This is the simplest adapter that requires no trained model. It ranks
/// candidates by comparing field values in the order specified by
/// static_order_fields (e.g., ["priority", "id"]).
///
/// The "score" is computed such that earlier fields have higher weight,
/// and lower field values are ranked higher (for "max" objective).
class StaticOrderAdapter : public IUhdAdapter
{
public:
    /// Construct with ordering fields and their indices in the feature vector.
    /// @param orderFieldIndices Indices of fields to sort by (in priority order).
    /// @param numFeatures Total number of features expected.
    StaticOrderAdapter(std::vector<size_t> orderFieldIndices, size_t numFeatures);

    /// Create from field names and a features signature.
    /// @param orderFields Field names to sort by (e.g., ["priority", "id"]).
    /// @param signature Features signature (to resolve field names to indices).
    /// @returns Adapter or nullptr if field names not found in signature.
    static std::unique_ptr<StaticOrderAdapter>
        create(const std::vector<std::string>& orderFields,
               const std::vector<std::string>& signature);

    double score(const std::vector<double>& features) const override;

    UhdAdapterType type() const override { return UhdAdapterType::StaticOrder; }

    size_t expectedFeatureCount() const override { return _numFeatures; }

    const std::string& getFeaturesHash() const override { return _emptyHash; }

private:
    std::vector<size_t> _orderFieldIndices;
    size_t _numFeatures;
    static const std::string _emptyHash;
};

} // namespace hipdnn_backend::heuristics::uhd
