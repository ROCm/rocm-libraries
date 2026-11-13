// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <stdexcept>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hipdnn_plugin_sdk::uhd
{

/// A catalog's feature rows, stored row-major in one buffer the caller owns.
struct FeatureRows
{
    const double* data = nullptr;
    size_t count = 0; // Rows.
    size_t width = 0; // Values per row; every row has the same width.

    const double* row(size_t index) const
    {
        return data + (index * width);
    }
};

/// What scoreBatch returns: one raw score per row and, for a model with an early-exit
/// schedule, how far each row got.
///
/// `stage` is empty when every row was scored by the whole model. Otherwise it holds one
/// entry per row: the number of early-exit stages the row survived. A row whose entry equals
/// `stages` was scored by the whole model and its `raw` is a full score. Any smaller entry
/// marks a row pruned at that stage: its `raw` is a partial sum over the trees it saw, in no
/// unit, comparable only with rows pruned at the same stage, in the objective's direction.
/// A ranker places every fully scored row ahead of every pruned row, and later stages ahead
/// of earlier ones.
struct BatchScores
{
    std::vector<double> raw;
    std::vector<uint8_t> stage;
    uint8_t stages = 0;

    /// True when this holds one raw score for each of @p rows rows and, if staged, one stage
    /// entry for each. A ranker checks this before indexing by row.
    bool describes(size_t rows) const
    {
        return raw.size() == rows && (stage.empty() || stage.size() == rows);
    }

    /// False for a row whose stage entry is missing: nothing shows the whole model scored it.
    bool fullyScored(size_t index) const
    {
        return stage.empty() || (index < stage.size() && stage[index] == stages);
    }
};

/// @brief Abstract interface for UHD model adapters: scores candidates from feature vectors.
class IUhdAdapter
{
public:
    virtual ~IUhdAdapter() = default;

    /// Score a single candidate.
    /// @param features Feature vector; size must equal expectedFeatureCount().
    /// @returns Predicted score; meaning depends on the UHD objective.
    virtual double score(const std::vector<double>& features) const = 0;

    /// Score a catalog's rows. The default calls score() per row; adapters whose model
    /// compares rows (a grouped model's group choice, an early-exit schedule) override it.
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    virtual BatchScores scoreBatch(const FeatureRows& rows) const
    {
        BatchScores results;
        results.raw.reserve(rows.count);
        std::vector<double> features(rows.width);
        for(size_t i = 0; i < rows.count; ++i)
        {
            features.assign(rows.row(i), rows.row(i) + rows.width);
            results.raw.push_back(score(features));
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

    /// Number of features a row must carry. Artifact-backed adapters (`tree_data`, `table`)
    /// read it from the artifact; `native` and `custom_library` report the descriptor's
    /// signature length, so for them it attests nothing (RFC 0019 OQ11).
    virtual size_t expectedFeatureCount() const = 0;

    /// Features hash the model was trained on. Artifact-backed adapters read it from the
    /// artifact and refuse to load on a mismatch with the descriptor; `native` and
    /// `custom_library` echo the descriptor's `features_hash`, which is trusted as declared.
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
