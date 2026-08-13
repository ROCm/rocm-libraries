// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TableAdapter.hpp"

#include <algorithm>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/table_model_generated.h>

namespace hipdnn_backend::heuristics::uhd
{

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

std::size_t VectorHash::operator()(const std::vector<uint32_t>& vec) const
{
    std::size_t seed = vec.size();
    for(auto val : vec)
    {
        // Hash combining from Boost
        seed ^= static_cast<std::size_t>(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

std::unique_ptr<TableAdapter> TableAdapter::load(const std::string& modelPath,
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

std::unique_ptr<TableAdapter> TableAdapter::loadFromBuffer(const uint8_t* buffer,
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

TableAdapter::TableAdapter(std::vector<uint8_t> ownedBuffer,
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

TableAdapter::~TableAdapter() = default;

uint32_t TableAdapter::quantize(double value, const std::vector<double>& boundaries)
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

std::vector<uint32_t> TableAdapter::buildBucketKey(const std::vector<double>& features) const
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

double TableAdapter::score(const std::vector<double>& features) const
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

bool TableAdapter::isTrainedForArch(const std::string& arch) const
{
    // If no training arches specified, assume works for all
    if(_trainingArches.empty())
    {
        return true;
    }

    // Check if arch is in the training set
    return std::find(_trainingArches.begin(), _trainingArches.end(), arch) != _trainingArches.end();
}

} // namespace hipdnn_backend::heuristics::uhd
