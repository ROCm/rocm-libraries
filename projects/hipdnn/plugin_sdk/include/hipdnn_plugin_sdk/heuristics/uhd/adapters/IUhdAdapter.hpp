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

/// @brief Abstract interface for UHD model adapters: scores candidates from feature vectors.
class IUhdAdapter
{
public:
    virtual ~IUhdAdapter() = default;

    /// Score a single candidate.
    /// @param features Feature vector; size must equal expectedFeatureCount().
    /// @returns Predicted score; meaning depends on the UHD objective.
    virtual double score(const std::vector<double>& features) const = 0;

    /// Score multiple candidates; the default calls score() per row.
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

    /// Feature-row slot holding a candidate's group, or -1 if this adapter is not two-layer.
    /// Lets a ranker report each candidate's group exactly as scoreBatch decided it.
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    virtual int groupFeatureIndex() const
    {
        return -1;
    }

    /// Expected number of features.
    virtual size_t expectedFeatureCount() const = 0;

    /// Features hash this adapter was trained on (for contract validation).
    virtual const std::string& getFeaturesHash() const = 0;

    /// True if @p arch was seen in training or training_arches is empty.
    /// The default imposes no restriction.
    virtual bool isTrainedForArch(const std::string& /*arch*/) const
    {
        return true;
    }
};

} // namespace hipdnn_plugin_sdk::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
