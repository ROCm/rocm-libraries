// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "IUhdAdapter.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <fstream>
#include <functional>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/table_model_generated.h>
#include <sstream>
#include <stdexcept>

// Forward declare FlatBuffer types
namespace hipdnn_flatbuffers_sdk::data_objects
{
struct TableModel;
struct FeatureBucket;
} // namespace hipdnn_flatbuffers_sdk::data_objects

namespace hipdnn_plugin_sdk::ingestor::uhd
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

    ~TableAdapter() override = default;

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


namespace fb = hipdnn_flatbuffers_sdk::data_objects;

inline std::size_t VectorHash::operator()(const std::vector<uint32_t>& vec) const
{
    std::size_t seed = vec.size();
    for(auto val : vec)
    {
        // Hash combining from Boost
        seed ^= static_cast<std::size_t>(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

inline std::unique_ptr<TableAdapter> TableAdapter::load(const std::string& modelPath,
                                                 const std::string& expectedFeaturesHash)
{
    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    if(!file)
    {
        return nullptr;
    }

    auto size = file.tellg();
    if(size <= 0)
    {
        return nullptr;
    }

    std::vector<uint8_t> buffer(static_cast<size_t>(size));
    file.seekg(0);
    if(!file.read(reinterpret_cast<char*>(buffer.data()), size))
    {
        return nullptr;
    }

    return loadFromBuffer(buffer.data(), buffer.size(), expectedFeaturesHash);
}

inline std::unique_ptr<TableAdapter> TableAdapter::loadFromBuffer(const uint8_t* buffer,
                                                           size_t size,
                                                           const std::string& expectedFeaturesHash)
{
    // Guard against null/empty buffer
    if(buffer == nullptr || size < sizeof(flatbuffers::uoffset_t) + 4)
    {
        return nullptr;
    }

    // Verify file identifier
    if(!flatbuffers::BufferHasIdentifier(buffer, fb::TableModelIdentifier()))
    {
        return nullptr;
    }

    // Verify buffer
    flatbuffers::Verifier verifier(buffer, size);
    if(!fb::VerifyTableModelBuffer(verifier))
    {
        return nullptr;
    }

    const auto* model = fb::GetTableModel(buffer);
    if(model == nullptr)
    {
        return nullptr;
    }

    // Validate features hash
    const std::string modelHash
        = model->features_hash() != nullptr ? model->features_hash()->str() : "";
    if(!expectedFeaturesHash.empty() && modelHash != expectedFeaturesHash)
    {
        HIPDNN_SDK_LOG_WARN("TableAdapter: features hash mismatch - model='"
                            << modelHash << "' expected='" << expectedFeaturesHash << "'");
        return nullptr;
    }

    const auto numFeatures = static_cast<size_t>(model->num_features());

    // Extract training arches
    std::vector<std::string> trainingArches;
    if(model->training_arches() != nullptr)
    {
        for(const auto* arch : *model->training_arches())
        {
            if(arch != nullptr)
            {
                trainingArches.emplace_back(arch->str());
            }
        }
    }

    // Extract model version
    const std::string modelVersion
        = model->model_version() != nullptr ? model->model_version()->str() : "";

    // Copy buffer to owned storage
    std::vector<uint8_t> ownedBuffer(buffer, buffer + size);

    // Evaluate GetTableModel BEFORE moving ownedBuffer
    const fb::TableModel* modelPtr = fb::GetTableModel(ownedBuffer.data());
    return std::unique_ptr<TableAdapter>(new TableAdapter(std::move(ownedBuffer),
                                                          modelPtr,
                                                          modelHash,
                                                          numFeatures,
                                                          std::move(trainingArches),
                                                          modelVersion));
}

inline TableAdapter::TableAdapter(std::vector<uint8_t> ownedBuffer,
                           const fb::TableModel* model,
                           std::string featuresHash,
                           size_t numFeatures,
                           std::vector<std::string> trainingArches,
                           std::string modelVersion)
    : _ownedBuffer(std::move(ownedBuffer))
    , _model(model)
    , _featuresHash(std::move(featuresHash))
    , _numFeatures(numFeatures)
    , _trainingArches(std::move(trainingArches))
    , _modelVersion(std::move(modelVersion))
{
    // Build the lookup table from the model's entries
    if(_model->entries() != nullptr)
    {
        for(const auto* entry : *_model->entries())
        {
            if(entry == nullptr || entry->bucket_key() == nullptr)
            {
                continue;
            }

            // Convert FlatBuffers vector to std::vector
            std::vector<uint32_t> key;
            key.reserve(entry->bucket_key()->size());
            for(auto val : *entry->bucket_key())
            {
                key.push_back(val);
            }

            // Store in lookup table (use custom hash function)
            _lookupTable.emplace(std::move(key), entry->score());
        }
    }
}

inline uint32_t TableAdapter::quantize(double value, const std::vector<double>& boundaries)
{
    if(boundaries.empty())
    {
        return 0;
    }

    // Find the first boundary > value
    auto it = std::upper_bound(boundaries.begin(), boundaries.end(), value);

    // Return the bucket index (distance from begin)
    return static_cast<uint32_t>(std::distance(boundaries.begin(), it));
}

inline std::vector<uint32_t> TableAdapter::buildBucketKey(const std::vector<double>& features) const
{
    if(_model == nullptr || _model->buckets() == nullptr)
    {
        return {};
    }

    std::vector<uint32_t> key;
    key.reserve(_model->buckets()->size());

    for(const auto* bucket : *_model->buckets())
    {
        if(bucket == nullptr)
        {
            return {}; // Invalid bucket definition
        }

        const uint32_t featureIdx = bucket->feature_index();
        if(featureIdx >= features.size())
        {
            return {}; // Feature index out of range
        }

        // Extract boundaries
        std::vector<double> boundaries;
        if(bucket->boundaries() != nullptr)
        {
            boundaries.reserve(bucket->boundaries()->size());
            for(auto boundary : *bucket->boundaries())
            {
                boundaries.push_back(boundary);
            }
        }

        // Quantize the feature value
        const uint32_t bucketIdx = quantize(features[featureIdx], boundaries);
        key.push_back(bucketIdx);
    }

    return key;
}

inline double TableAdapter::score(const std::vector<double>& features) const
{
    // Build the bucket key from features
    const auto key = buildBucketKey(features);
    if(key.empty())
    {
        // Bucketing failed (invalid model or feature vector)
        return 0.0;
    }

    // Lookup in the prebuilt table
    const auto it = _lookupTable.find(key);
    if(it != _lookupTable.end())
    {
        return it->second;
    }

    // No exact match - return 0.0 (fallback to priority ordering)
    return 0.0;
}

inline bool TableAdapter::isTrainedForArch(const std::string& arch) const
{
    // If no training arches specified, assume works for all
    if(_trainingArches.empty())
    {
        return true;
    }

    // Check if arch is in the training set
    return std::find(_trainingArches.begin(), _trainingArches.end(), arch) != _trainingArches.end();
}

} // namespace hipdnn_plugin_sdk::ingestor::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
