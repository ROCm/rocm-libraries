// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TreeDataAdapter.hpp"

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
    if(!model)
    {
        return nullptr;
    }

    // Validate features hash
    std::string modelHash = model->features_hash() ? model->features_hash()->str() : "";
    if(!expectedFeaturesHash.empty() && modelHash != expectedFeaturesHash)
    {
        return nullptr;
    }

    size_t numFeatures = static_cast<size_t>(model->num_features());
    double baseScore = model->base_score();
    double learningRate = model->learning_rate();

    // Copy buffer to owned storage
    std::vector<uint8_t> ownedBuffer(buffer, buffer + size);

    return std::unique_ptr<TreeDataAdapter>(new TreeDataAdapter(std::move(ownedBuffer),
                                                                 fb::GetGbdtModel(ownedBuffer.data()),
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
      _treeCount(model->trees() ? model->trees()->size() : 0),
      _baseScore(baseScore),
      _learningRate(learningRate)
{
}

TreeDataAdapter::~TreeDataAdapter() = default;

double TreeDataAdapter::score(const std::vector<double>& features) const
{
    if(!_model || !_model->trees())
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
                                      const std::vector<double>& features) const
{
    if(!tree || !tree->feature_indices() || !tree->thresholds() || !tree->left_children() ||
       !tree->right_children() || !tree->leaf_values())
    {
        return 0.0;
    }

    const auto& featureIndices = *tree->feature_indices();
    const auto& thresholds = *tree->thresholds();
    const auto& leftChildren = *tree->left_children();
    const auto& rightChildren = *tree->right_children();
    const auto& leafValues = *tree->leaf_values();

    bool hasDefaultLeft = tree->default_left() && tree->default_left()->size() > 0;

    size_t node = 0;
    while(node < leftChildren.size())
    {
        int leftChild = leftChildren[node];
        int rightChild = rightChildren[node];

        // Leaf node: left_children[node] == -1
        if(leftChild < 0)
        {
            if(node < leafValues.size())
            {
                return leafValues[node];
            }
            return 0.0;
        }

        // Internal node: compare feature value against threshold
        int featureIdx = featureIndices[node];
        double threshold = thresholds[node];

        bool goLeft = false;
        if(featureIdx >= 0 && static_cast<size_t>(featureIdx) < features.size())
        {
            double featureVal = features[static_cast<size_t>(featureIdx)];

            // Check for NaN (missing value)
            if(std::isnan(featureVal))
            {
                goLeft = hasDefaultLeft && node < tree->default_left()->size() &&
                         (*tree->default_left())[node];
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
                     (*tree->default_left())[node];
        }

        node = static_cast<size_t>(goLeft ? leftChild : rightChild);
    }

    return 0.0;
}

} // namespace hipdnn_backend::heuristics::uhd
