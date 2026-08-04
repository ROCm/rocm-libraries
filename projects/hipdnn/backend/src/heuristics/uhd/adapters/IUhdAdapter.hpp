// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace hipdnn_backend::heuristics::uhd
{

/// @brief Adapter types matching UhdAdapter enum in uhd.fbs.
enum class UhdAdapterType
{
    STATIC_ORDER = 0,
    TREE_DATA = 1,
    TABLE = 2,
    ONNX = 3,
    CUSTOM_LIBRARY = 4,
};

/// @brief Abstract interface for UHD model adapters.
///
/// An adapter scores kernel candidates based on their feature vectors.
/// Different adapters support different model formats (GBDT, ONNX, etc.).
class IUhdAdapter
{
public:
    virtual ~IUhdAdapter() = default;

    /// Score a single candidate.
    /// @param features Feature vector (must match expected count).
    /// @returns Predicted score (interpretation depends on UHD objective).
    virtual double score(const std::vector<double>& features) const = 0;

    /// Score multiple candidates in batch.
    /// Default implementation calls score() for each row.
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    virtual std::vector<double> scoreBatch(const std::vector<std::vector<double>>& batch) const
    {
        std::vector<double> results;
        results.reserve(batch.size());
        for(const auto& features : batch)
        {
            results.push_back(score(features));
        }
        return results;
    }

    /// Get the adapter type.
    virtual UhdAdapterType type() const = 0;

    /// Get the expected number of features.
    virtual size_t expectedFeatureCount() const = 0;

    /// Validate that a feature vector has the expected count.
    bool validateFeatureCount(size_t count) const { return count == expectedFeatureCount(); }

    /// Get the features hash this adapter was trained on (for contract validation).
    virtual const std::string& getFeaturesHash() const = 0;
};

/// @brief Factory function type for creating adapters.
using AdapterFactory =
    std::unique_ptr<IUhdAdapter> (*)(const std::string& artifactPath,
                                     const std::string& expectedFeaturesHash);

} // namespace hipdnn_backend::heuristics::uhd
