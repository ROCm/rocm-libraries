// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <stdexcept>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace hipdnn_plugin_sdk::uhd
{

/// @brief Abstract interface for UHD model adapters.
///
/// An adapter scores kernel candidates based on their feature vectors.
/// Different adapters support different model formats (GBDT trees, lookup tables, native scorers).
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

    /// Which slot of the feature row identifies a candidate's group, or -1 when this adapter
    /// does not decide in two layers.
    ///
    /// `scoreBatch` already makes the group decision internally; this exposes only *where* the
    /// group is read from, so a ranker can report which group each candidate belonged to without
    /// re-deriving it from anywhere else and risking a different answer.
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    virtual int groupFeatureIndex() const
    {
        return -1;
    }

    /// Get the expected number of features.
    virtual size_t expectedFeatureCount() const = 0;

    /// Get the features hash this adapter was trained on (for contract validation).
    virtual const std::string& getFeaturesHash() const = 0;

    /// Check if the given architecture was seen during training.
    /// Returns true if training_arches is empty (no restriction) or if arch is in the list.
    /// Default implementation always returns true (no restriction).
    virtual bool isTrainedForArch(const std::string& /*arch*/) const
    {
        return true;
    }
};

} // namespace hipdnn_plugin_sdk::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
