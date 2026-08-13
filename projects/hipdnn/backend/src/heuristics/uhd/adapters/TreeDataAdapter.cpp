// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TreeDataAdapter.hpp"
#include "../Sha256.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/gbdt_model_generated.h>

namespace hipdnn_backend::heuristics::uhd
{

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

std::unique_ptr<TreeDataAdapter> TreeDataAdapter::load(const std::string& modelPath,
                                                       const std::string& expectedFeaturesHash,
                                                       const std::string& expectedModelHash)
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

    return loadFromBuffer(buffer.data(), buffer.size(), expectedFeaturesHash, expectedModelHash);
}

std::unique_ptr<TreeDataAdapter> TreeDataAdapter::loadFromBuffer(const uint8_t* buffer,
                                                                 size_t size,
                                                                 const std::string& expectedFeaturesHash,
                                                                 const std::string& expectedModelHash)
{
    // Guard against null/empty buffer
    if(buffer == nullptr || size < sizeof(flatbuffers::uoffset_t) + 4)
    {
        return nullptr;
    }

    // Validate model hash if provided (RFC 0019 §9.2 integrity validation)
    if(!expectedModelHash.empty())
    {
        const std::string actualHash = sha256(buffer, size);
        if(actualHash != expectedModelHash)
        {
            HIPDNN_SDK_LOG_WARN("TreeDataAdapter: model hash mismatch - computed='"
                                << actualHash << "' expected='" << expectedModelHash << "'");
            return nullptr;
        }
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
    const std::string modelHash
        = model->features_hash() != nullptr ? model->features_hash()->str() : "";
    if(!expectedFeaturesHash.empty() && modelHash != expectedFeaturesHash)
    {
        HIPDNN_SDK_LOG_WARN("TreeDataAdapter: features hash mismatch - model='"
                            << modelHash << "' expected='" << expectedFeaturesHash << "'");
        return nullptr;
    }

    const auto numFeatures = static_cast<size_t>(model->num_features());
    const double baseScore = model->base_score();
    const double learningRate = model->learning_rate();

    // Extract training arches (RFC 0019 §9.2)
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

    // Extract model version (RFC 0019 §13)
    const std::string modelVersion
        = model->model_version() != nullptr ? model->model_version()->str() : "";

    // Copy buffer to owned storage
    std::vector<uint8_t> ownedBuffer(buffer, buffer + size);

    // Evaluate GetGbdtModel BEFORE moving ownedBuffer to avoid unspecified
    // argument evaluation order issues (the model pointer must be obtained
    // while the buffer is still valid)
    const fb::GbdtModel* modelPtr = fb::GetGbdtModel(ownedBuffer.data());
    return std::unique_ptr<TreeDataAdapter>(new TreeDataAdapter(std::move(ownedBuffer),
                                                                modelPtr,
                                                                modelHash,
                                                                numFeatures,
                                                                baseScore,
                                                                learningRate,
                                                                std::move(trainingArches),
                                                                modelVersion));
}

TreeDataAdapter::TreeDataAdapter(std::vector<uint8_t> ownedBuffer,
                                 const fb::GbdtModel* model,
                                 std::string featuresHash,
                                 size_t numFeatures,
                                 double baseScore,
                                 double learningRate,
                                 std::vector<std::string> trainingArches,
                                 std::string modelVersion)
    : _ownedBuffer(std::move(ownedBuffer))
    , _model(model)
    , _featuresHash(std::move(featuresHash))
    , _numFeatures(numFeatures)
    , _treeCount(model->trees() != nullptr ? model->trees()->size() : 0)
    , _baseScore(baseScore)
    , _learningRate(learningRate)
    , _trainingArches(std::move(trainingArches))
    , _modelVersion(std::move(modelVersion))
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

    // NOTE: LightGBM's dump_model() returns leaf_values that already include
    // the learning_rate multiplication. The raw_score prediction is:
    //   base_score + sum(leaf_values)
    // We do NOT multiply by learning_rate again here.
    // The _learningRate field is kept for metadata/documentation purposes only.
    return _baseScore + sum;
}

double TreeDataAdapter::evaluateTree(const fb::GbdtTree* tree, const std::vector<double>& features)
{
    if(tree == nullptr || tree->feature_indices() == nullptr || tree->thresholds() == nullptr
       || tree->left_children() == nullptr || tree->right_children() == nullptr
       || tree->leaf_values() == nullptr)
    {
        return 0.0;
    }

    const auto& featureIndices = *tree->feature_indices();
    const auto& thresholds = *tree->thresholds();
    const auto& leftChildren = *tree->left_children();
    const auto& rightChildren = *tree->right_children();
    const auto& leafValues = *tree->leaf_values();

    const bool hasDefaultLeft = tree->default_left() != nullptr && !tree->default_left()->empty();

    // Bound the descent (RFC 0019 §16: the model artifact is author-controlled input,
    // so the evaluator must be bounded). FlatBuffers' Verifier checks buffer layout,
    // not graph acyclicity — a tree whose child index points at an ancestor, or at
    // itself, is a well-formed buffer that would spin here forever and hang plan build.
    // A well-formed descent visits each node at most once, so exceeding the node count
    // means the tree is cyclic.
    const size_t maxSteps = leftChildren.size();
    size_t steps = 0;

    size_t node = 0;
    while(node < leftChildren.size())
    {
        if(++steps > maxSteps)
        {
            throw std::runtime_error(
                "TreeDataAdapter: tree descent exceeded " + std::to_string(maxSteps)
                + " steps, so the model artifact contains a cycle in its child indices");
        }

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

        // Check decision type: if decision_lte is present and true (or absent, defaulting to true),
        // use <= comparison (LightGBM standard). Otherwise use < (strict less-than).
        const bool hasDecisionLte
            = tree->decision_lte() != nullptr && !tree->decision_lte()->empty();
        const bool useLte
            = !hasDecisionLte
              || (node < tree->decision_lte()->size() && ((*tree->decision_lte())[nodeIdx] != 0u));

        bool goLeft = false;
        if(featureIdx >= 0 && static_cast<size_t>(featureIdx) < features.size())
        {
            const double featureVal = features[static_cast<size_t>(featureIdx)];

            // Check for NaN (missing value)
            if(std::isnan(featureVal))
            {
                goLeft = hasDefaultLeft && node < tree->default_left()->size()
                         && ((*tree->default_left())[nodeIdx] != 0u);
            }
            else
            {
                goLeft = useLte ? (featureVal <= threshold) : (featureVal < threshold);
            }
        }
        else
        {
            // Out of range feature index: use default direction
            goLeft = hasDefaultLeft && node < tree->default_left()->size()
                     && ((*tree->default_left())[nodeIdx] != 0u);
        }

        node = static_cast<size_t>(goLeft ? leftChild : rightChild);
    }

    return 0.0;
}

bool TreeDataAdapter::isTrainedForArch(const std::string& arch) const
{
    // If no training arches specified, assume the model works for all arches
    if(_trainingArches.empty())
    {
        return true;
    }

    // Check if the given arch is in the training set
    return std::find(_trainingArches.begin(), _trainingArches.end(), arch) != _trainingArches.end();
}

} // namespace hipdnn_backend::heuristics::uhd
