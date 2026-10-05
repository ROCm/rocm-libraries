// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "IUhdAdapter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/table_model_generated.h>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/Sha256.hpp>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declare FlatBuffer types
namespace hipdnn_flatbuffers_sdk::data_objects
{
struct TableModel;
struct FeatureBucket;
} // namespace hipdnn_flatbuffers_sdk::data_objects

namespace hipdnn_plugin_sdk::uhd
{

/// Hash function for std::vector<uint32_t> keys in the lookup table.
struct VectorHash
{
    std::size_t operator()(const std::vector<uint32_t>& vec) const;
};

/// @brief Table-based lookup adapter for coarse problem buckets (RFC 0019 §7 "table").
///
/// Quantizes features into buckets and looks the bucket combination up in a precomputed
/// table. Uncovered rows score -infinity (declined), so they rank last under either objective.
class TableAdapter : public IUhdAdapter
{
public:
    /// Load a table model from a FlatBuffer file.
    /// @param expectedModelHash Artifact SHA-256 hex; empty skips the check.
    /// @returns nullptr if loading or validation fails.
    static std::unique_ptr<TableAdapter> load(const std::string& modelPath,
                                              const std::string& expectedFeaturesHash,
                                              const std::string& expectedModelHash = "");

    /// Load from an in-memory buffer, which is copied into the adapter.
    /// @param expectedModelHash Artifact SHA-256 hex; empty skips the check.
    /// @returns nullptr if validation fails.
    static std::unique_ptr<TableAdapter> loadFromBuffer(const uint8_t* buffer,
                                                        size_t size,
                                                        const std::string& expectedFeaturesHash,
                                                        const std::string& expectedModelHash = "");

    ~TableAdapter() override = default;

    /// Non-copyable: `_model` points into `_ownedBuffer`, so copying would dangle.
    TableAdapter(const TableAdapter&) = delete;
    TableAdapter& operator=(const TableAdapter&) = delete;

    /// @returns The table score for the features' bucket, or -infinity (declined) if absent.
    double score(const std::vector<double>& features) const override;

    size_t expectedFeatureCount() const override
    {
        return _numFeatures;
    }

    const std::string& getFeaturesHash() const override
    {
        return _featuresHash;
    }

    bool isTrainedForArch(const std::string& arch) const override;

private:
    using LookupTable = std::unordered_map<std::vector<uint32_t>, double, VectorHash>;

    TableAdapter(std::vector<uint8_t> ownedBuffer,
                 const hipdnn_flatbuffers_sdk::data_objects::TableModel* model,
                 std::string featuresHash,
                 size_t numFeatures,
                 std::vector<std::string> trainingArches,
                 LookupTable lookupTable);

    /// Checks what the verifier cannot (table_model.fbs) and indexes the entries.
    /// @returns Why @p model is unusable, or empty with @p table filled.
    static std::string
        buildLookupTable(const hipdnn_flatbuffers_sdk::data_objects::TableModel& model,
                         size_t numFeatures,
                         LookupTable& table);

    /// @returns The number of @p boundaries <= @p value, in [0, boundaries.size()].
    static uint32_t quantize(double value, const flatbuffers::Vector<double>* boundaries);

    /// @returns One bucket index per bucketed feature, or empty if bucketing fails.
    std::vector<uint32_t> buildBucketKey(const std::vector<double>& features) const;

    std::vector<uint8_t> _ownedBuffer;
    const hipdnn_flatbuffers_sdk::data_objects::TableModel* _model;
    std::string _featuresHash;
    size_t _numFeatures;
    std::vector<std::string> _trainingArches;

    /// bucket_key -> score.
    LookupTable _lookupTable;
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
                                                        const std::string& expectedFeaturesHash,
                                                        const std::string& expectedModelHash)
{
    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    if(!file)
    {
        return nullptr;
    }

    auto size = file.tellg();
    if(size <= 0 || size > static_cast<std::streamoff>(256 * 1024 * 1024))
    {
        return nullptr;
    }

    std::vector<uint8_t> buffer(static_cast<size_t>(size));
    file.seekg(0);
    if(!file.read(reinterpret_cast<char*>(buffer.data()), size))
    {
        return nullptr;
    }

    return loadFromBuffer(buffer.data(), buffer.size(), expectedFeaturesHash, expectedModelHash);
}

inline std::unique_ptr<TableAdapter>
    TableAdapter::loadFromBuffer(const uint8_t* buffer,
                                 size_t size,
                                 const std::string& expectedFeaturesHash,
                                 const std::string& expectedModelHash)
{
    if(buffer == nullptr || size < sizeof(flatbuffers::uoffset_t) + 4
       || size > size_t{256} * 1024 * 1024)
    {
        return nullptr;
    }

    // RFC 0019 §9.2: the digest identifies the model to persistent caches, so mismatched
    // bytes must not be scored under it.
    if(!expectedModelHash.empty())
    {
        const std::string actualHash = sha256(buffer, size);
        if(actualHash != expectedModelHash)
        {
            HIPDNN_SDK_LOG_ERROR(
                "TableAdapter: model hash mismatch - expected='"
                << expectedModelHash << "' actual='" << actualHash
                << "'; the model is not used -- ranking degrades to static_order and an "
                   "engine estimate is reported as 0");
            return nullptr;
        }
    }

    if(!flatbuffers::BufferHasIdentifier(buffer, fb::TableModelIdentifier()))
    {
        return nullptr;
    }

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

    // RFC 0019 §6.3 check 3. ERROR, not WARN (§12), worded like the other adapters so one
    // log search finds all of them.
    const std::string modelHash
        = model->features_hash() != nullptr ? model->features_hash()->str() : "";
    if(!expectedFeaturesHash.empty() && modelHash != expectedFeaturesHash)
    {
        HIPDNN_SDK_LOG_ERROR("TableAdapter: features hash mismatch - expected='"
                             << expectedFeaturesHash << "' actual='" << modelHash
                             << "'; the model is not used -- ranking degrades to static_order "
                                "and an engine estimate is reported as 0");
        return nullptr;
    }

    const auto numFeatures = static_cast<size_t>(model->num_features());

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

    LookupTable lookupTable;
    const auto reason = buildLookupTable(*model, numFeatures, lookupTable);
    if(!reason.empty())
    {
        HIPDNN_SDK_LOG_ERROR("TableAdapter: " << reason
                                              << "; the model is not used -- ranking degrades to "
                                                 "static_order and an engine estimate is reported "
                                                 "as 0");
        return nullptr;
    }

    std::vector<uint8_t> ownedBuffer(buffer, buffer + size);

    // Evaluate GetTableModel BEFORE moving ownedBuffer
    const fb::TableModel* modelPtr = fb::GetTableModel(ownedBuffer.data());
    return std::unique_ptr<TableAdapter>(new TableAdapter(std::move(ownedBuffer),
                                                          modelPtr,
                                                          modelHash,
                                                          numFeatures,
                                                          std::move(trainingArches),
                                                          std::move(lookupTable)));
}

inline TableAdapter::TableAdapter(std::vector<uint8_t> ownedBuffer,
                                  const fb::TableModel* model,
                                  std::string featuresHash,
                                  size_t numFeatures,
                                  std::vector<std::string> trainingArches,
                                  LookupTable lookupTable)
    : _ownedBuffer(std::move(ownedBuffer))
    , _model(model)
    , _featuresHash(std::move(featuresHash))
    , _numFeatures(numFeatures)
    , _trainingArches(std::move(trainingArches))
    , _lookupTable(std::move(lookupTable))
{
}

inline std::string TableAdapter::buildLookupTable(const fb::TableModel& model,
                                                  size_t numFeatures,
                                                  LookupTable& table)
{
    const auto* buckets = model.buckets();
    if(buckets == nullptr || buckets->empty())
    {
        return "the model defines no buckets";
    }

    // Bucket i's indices run 0..bucketCounts[i]-1.
    std::vector<size_t> bucketCounts;
    bucketCounts.reserve(buckets->size());
    for(flatbuffers::uoffset_t i = 0; i < buckets->size(); ++i)
    {
        const auto* bucket = buckets->Get(i);
        if(bucket == nullptr)
        {
            return "bucket " + std::to_string(i) + " is missing";
        }
        if(bucket->feature_index() >= numFeatures)
        {
            return "bucket " + std::to_string(i) + " reads feature "
                   + std::to_string(bucket->feature_index()) + " of a model with "
                   + std::to_string(numFeatures);
        }

        const auto* boundaries = bucket->boundaries();
        const size_t boundaryCount = boundaries != nullptr ? boundaries->size() : 0;
        for(size_t b = 0; b < boundaryCount; ++b)
        {
            const double boundary = boundaries->Get(static_cast<flatbuffers::uoffset_t>(b));
            if(!std::isfinite(boundary)
               || (b > 0
                   && !(boundaries->Get(static_cast<flatbuffers::uoffset_t>(b - 1)) < boundary)))
            {
                return "bucket " + std::to_string(i)
                       + "'s boundaries are not finite and strictly ascending";
            }
        }
        bucketCounts.push_back(boundaryCount + 1);
    }

    if(model.entries() == nullptr)
    {
        return {};
    }

    table.reserve(model.entries()->size());
    for(flatbuffers::uoffset_t e = 0; e < model.entries()->size(); ++e)
    {
        const auto* entry = model.entries()->Get(e);
        const auto* bucketKey = entry != nullptr ? entry->bucket_key() : nullptr;
        if(bucketKey == nullptr || bucketKey->size() != bucketCounts.size())
        {
            return "entry " + std::to_string(e)
                   + "'s bucket_key does not have one index per bucket";
        }

        std::vector<uint32_t> key(bucketKey->begin(), bucketKey->end());
        for(size_t i = 0; i < key.size(); ++i)
        {
            if(key[i] >= bucketCounts[i])
            {
                return "entry " + std::to_string(e) + "'s bucket_key names bucket "
                       + std::to_string(key[i]) + " of bucket " + std::to_string(i) + ", which has "
                       + std::to_string(bucketCounts[i]);
            }
        }

        if(!std::isfinite(entry->score()))
        {
            return "entry " + std::to_string(e) + "'s score is not finite";
        }

        if(!table.emplace(std::move(key), entry->score()).second)
        {
            return "entry " + std::to_string(e) + " repeats an earlier entry's bucket_key";
        }
    }

    return {};
}

inline uint32_t TableAdapter::quantize(double value, const flatbuffers::Vector<double>* boundaries)
{
    if(boundaries == nullptr)
    {
        return 0;
    }

    const auto it = std::upper_bound(boundaries->begin(), boundaries->end(), value);
    return static_cast<uint32_t>(it - boundaries->begin());
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

        const uint32_t bucketIdx = quantize(features[featureIdx], bucket->boundaries());
        key.push_back(bucketIdx);
    }

    return key;
}

inline double TableAdapter::score(const std::vector<double>& features) const
{
    const auto key = buildBucketKey(features);
    if(key.empty())
    {
        // Bucketing failed (invalid model or feature vector)
        return -std::numeric_limits<double>::infinity();
    }

    const auto it = _lookupTable.find(key);
    if(it != _lookupTable.end())
    {
        return it->second;
    }

    // No match: decline. 0.0 would look like a real prediction and, under `objective: min`,
    // outrank every covered candidate.
    return -std::numeric_limits<double>::infinity();
}

inline bool TableAdapter::isTrainedForArch(const std::string& arch) const
{
    if(_trainingArches.empty())
    {
        return true;
    }

    const auto target = stripArchFeatures(arch);
    return std::find(_trainingArches.begin(), _trainingArches.end(), target)
           != _trainingArches.end();
}

} // namespace hipdnn_plugin_sdk::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
