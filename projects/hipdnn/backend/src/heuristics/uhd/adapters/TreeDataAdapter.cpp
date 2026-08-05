// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TreeDataAdapter.hpp"

#include <cmath>
#include <fstream>

#include <hipdnn_flatbuffers_sdk/data_objects/gbdt_model_generated.h>

namespace hipdnn_backend::heuristics::uhd
{

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

std::unique_ptr<TreeDataAdapter> TreeDataAdapter::load(const std::string& modelPath,
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

std::unique_ptr<TreeDataAdapter>
    TreeDataAdapter::loadFromBuffer(const uint8_t* buffer,
                                    size_t size,
                                    const std::string& expectedFeaturesHash)
{
    // Guard against null/empty buffer
    if(buffer == nullptr || size < sizeof(flatbuffers::uoffset_t) + 4)
    {
        return nullptr;
    }

    // Verify file identifier
    if(!flatbuffers::BufferHasIdentifier(buffer, fb::GbdtModelIdentifier()))
    {
        return nullptr;
    }

    // Verify buffer
    flatbuffers::Verifier verifier(buffer, size);
    if(!fb::VerifyGbdtModelBuffer(verifier))
    {
        return nullptr;
    }

    const auto* model = fb::GetGbdtModel(buffer);
    if(model == nullptr)
    {
        return nullptr;
    }

    // Validate features hash
    const std::string modelHash =
        model->features_hash() != nullptr ? model->features_hash()->str() : "";
    if(!expectedFeaturesHash.empty() && modelHash != expectedFeaturesHash)
    {
        return nullptr;
    }

    const auto numFeatures = static_cast<size_t>(model->num_features());
    const double baseScore = model->base_score();
    const double learningRate = model->learning_rate();

    // Copy buffer to owned storage
    std::vector<uint8_t> ownedBuffer(buffer, buffer + size);

    // Create the adapter with the owned buffer data pointer before moving
    const uint8_t* bufferData = ownedBuffer.data();
    return std::unique_ptr<TreeDataAdapter>(new TreeDataAdapter(std::move(ownedBuffer),
                                                                 fb::GetGbdtModel(bufferData),
                                                                 modelHash,
                                                                 numFeatures,
                                                                 baseScore,
                                                                 learningRate));
}

TreeDataAdapter::TreeDataAdapter(std::vector<uint8_t> ownedBuffer,
                                  const fb::GbdtModel* model,
                                  std::string featuresHash,
                                  size_t numFeatures,
                                  double baseScore,
                                  double learningRate)
    : _ownedBuffer(std::move(ownedBuffer)),
      _model(model),
      _featuresHash(std::move(featuresHash)),
      _numFeatures(numFeatures),
      _treeCount(model->trees() != nullptr ? model->trees()->size() : 0),
      _baseScore(baseScore),
      _learningRate(learningRate)
{
}

TreeDataAdapter::~TreeDataAdapter() = default;

double TreeDataAdapter::score(const std::vector<double>& features) const
{
    if(_model == nullptr || _model->trees() == nullptr)
    {
        return _baseScore;
    }

    double sum = 0.0;
    for(const auto* tree : *_model->trees())
    {
        sum += evaluateTree(tree, features);
    }

    return _baseScore + _learningRate * sum;
}

double TreeDataAdapter::evaluateTree(const fb::GbdtTree* tree,
                                      const std::vector<double>& features)
{
    if(tree == nullptr || tree->feature_indices() == nullptr || tree->thresholds() == nullptr ||
       tree->left_children() == nullptr || tree->right_children() == nullptr ||
       tree->leaf_values() == nullptr)
    {
        return 0.0;
    }

    const auto& featureIndices = *tree->feature_indices();
    const auto& thresholds = *tree->thresholds();
    const auto& leftChildren = *tree->left_children();
    const auto& rightChildren = *tree->right_children();
    const auto& leafValues = *tree->leaf_values();

    const bool hasDefaultLeft = tree->default_left() != nullptr && !tree->default_left()->empty();

    size_t node = 0;
    while(node < leftChildren.size())
    {
        const auto nodeIdx = static_cast<flatbuffers::uoffset_t>(node);
        const int leftChild = leftChildren[nodeIdx];
        const int rightChild = rightChildren[nodeIdx];

        // Leaf node: left_children[node] == -1
        if(leftChild < 0)
        {
            if(node < leafValues.size())
            {
                return leafValues[nodeIdx];
            }
            return 0.0;
        }

        // Internal node: compare feature value against threshold
        const int featureIdx = featureIndices[nodeIdx];
        const double threshold = thresholds[nodeIdx];

        bool goLeft = false;
        if(featureIdx >= 0 && static_cast<size_t>(featureIdx) < features.size())
        {
            const double featureVal = features[static_cast<size_t>(featureIdx)];

            // Check for NaN (missing value)
            if(std::isnan(featureVal))
            {
                goLeft = hasDefaultLeft && node < tree->default_left()->size() &&
                         ((*tree->default_left())[nodeIdx] != 0u);
            }
            else
            {
                goLeft = featureVal < threshold;
            }
        }
        else
        {
            // Out of range feature index: use default direction
            goLeft = hasDefaultLeft && node < tree->default_left()->size() &&
                     ((*tree->default_left())[nodeIdx] != 0u);
        }

        node = static_cast<size_t>(goLeft ? leftChild : rightChild);
    }

    return 0.0;
}

} // namespace hipdnn_backend::heuristics::uhd
