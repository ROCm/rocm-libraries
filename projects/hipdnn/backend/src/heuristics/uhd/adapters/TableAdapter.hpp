// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "IUhdAdapter.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declare FlatBuffer types
namespace hipdnn_flatbuffers_sdk::data_objects
{
struct TableModel;
struct FeatureBucket;
} // namespace hipdnn_flatbuffers_sdk::data_objects

namespace hipdnn_backend::heuristics::uhd
{

/// Hash function for std::vector<uint32_t> keys in the lookup table.
struct VectorHash
{
    std::size_t operator()(const std::vector<uint32_t>& vec) const;
};

/// @brief Table-based lookup adapter for coarse problem buckets (RFC 0019 §7 "table").
///
/// Maps feature vectors to kernel IDs via bucketing and table lookup. Features are
/// quantized into discrete buckets, then the bucket combination is looked up in a
/// precomputed table. Falls back to score=0.0 when no exact match exists (caller
/// then uses priority/id ordering).
class TableAdapter : public IUhdAdapter
{
public:
    /// Load a table model from a FlatBuffer file.
    /// @param modelPath Path to the .fb model file.
    /// @param expectedFeaturesHash Hash from UHD features_signature.
    /// @returns Adapter or nullptr if loading/validation fails.
    static std::unique_ptr<TableAdapter> load(const std::string& modelPath,
                                              const std::string& expectedFeaturesHash);

    /// Load from an in-memory buffer.
    /// @param buffer FlatBuffer data. Copied into the adapter.
    /// @param size Size of buffer in bytes.
    /// @param expectedFeaturesHash Hash from UHD features_signature.
    /// @returns Adapter or nullptr if validation fails.
    static std::unique_ptr<TableAdapter> loadFromBuffer(const uint8_t* buffer,
                                                        size_t size,
                                                        const std::string& expectedFeaturesHash);

    ~TableAdapter() override;

    /// Non-copyable: `_model` points into `_ownedBuffer`, so copying would dangle.
    TableAdapter(const TableAdapter&) = delete;
    TableAdapter& operator=(const TableAdapter&) = delete;

    /// Score a candidate by bucketing features and looking up in the table.
    /// @param features Feature vector (must match expected count).
    /// @returns Score from table if bucket match found, 0.0 otherwise (fallback to priority).
    double score(const std::vector<double>& features) const override;

    UhdAdapterType type() const override
    {
        return UhdAdapterType::TABLE;
    }

    size_t expectedFeatureCount() const override
    {
        return _numFeatures;
    }

    const std::string& getFeaturesHash() const override
    {
        return _featuresHash;
    }

    std::string getModelVersion() const override
    {
        return _modelVersion;
    }

    std::vector<std::string> getTrainingArches() const override
    {
        return _trainingArches;
    }

    bool isTrainedForArch(const std::string& arch) const override;

private:
    TableAdapter(std::vector<uint8_t> ownedBuffer,
                const hipdnn_flatbuffers_sdk::data_objects::TableModel* model,
                std::string featuresHash,
                size_t numFeatures,
                std::vector<std::string> trainingArches,
                std::string modelVersion);

    /// Quantize a feature value into a bucket index using the feature's boundaries.
    /// @param value Feature value to bucket.
    /// @param boundaries Sorted bucket boundaries.
    /// @returns Bucket index (0 to boundaries.size()).
    static uint32_t quantize(double value, const std::vector<double>& boundaries);

    /// Build bucket key from feature vector.
    /// @param features Input feature vector.
    /// @returns Bucket indices for each bucketed feature, or empty vector if bucketing fails.
    std::vector<uint32_t> buildBucketKey(const std::vector<double>& features) const;

    std::vector<uint8_t> _ownedBuffer;
    const hipdnn_flatbuffers_sdk::data_objects::TableModel* _model;
    std::string _featuresHash;
    size_t _numFeatures;
    std::vector<std::string> _trainingArches;
    std::string _modelVersion;

    /// Precomputed lookup table: bucket_key -> score.
    /// Built during construction from the model's entries.
    /// Uses custom hash function for vector<uint32_t> keys.
    std::unordered_map<std::vector<uint32_t>, double, VectorHash> _lookupTable;
};

} // namespace hipdnn_backend::heuristics::uhd
