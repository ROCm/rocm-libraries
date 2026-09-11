// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "IUhdAdapter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/gbdt_model_generated.h>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/Sha256.hpp>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace hipdnn_plugin_sdk::uhd
{

/// @brief GBDT tree walker adapter for scoring kernel candidates.
///
/// Validates a FlatBuffer model and prepares contiguous nodes once at load time.
/// The model's features_hash must match the UHD's signature hash.
class TreeDataAdapter : public IUhdAdapter
{
public:
    /// Load a GBDT model from a FlatBuffer file.
    /// @param modelPath Path to the .fb model file.
    /// @param expectedFeaturesHash Hash from UHD features_signature.
    /// @param expectedModelHash Optional checksum of the model file for integrity validation.
    /// @returns Adapter or nullptr if loading/validation fails.
    static std::unique_ptr<TreeDataAdapter> load(const std::string& modelPath,
                                                 const std::string& expectedFeaturesHash,
                                                 const std::string& expectedModelHash = "");

    /// Load from an in-memory buffer.
    /// @param buffer FlatBuffer data. Used only during this call; prepared nodes
    ///        and metadata are owned by the returned adapter.
    /// @param size Size of buffer in bytes.
    /// @param expectedFeaturesHash Hash from UHD features_signature.
    /// @param expectedModelHash Optional checksum of the model file for integrity validation.
    /// @returns Adapter or nullptr if validation fails.
    static std::unique_ptr<TreeDataAdapter> loadFromBuffer(const uint8_t* buffer,
                                                           size_t size,
                                                           const std::string& expectedFeaturesHash,
                                                           const std::string& expectedModelHash
                                                           = "");

    ~TreeDataAdapter() override = default;

    TreeDataAdapter(const TreeDataAdapter&) = delete;
    TreeDataAdapter& operator=(const TreeDataAdapter&) = delete;

    double score(const std::vector<double>& features) const override;

    UhdAdapterType type() const override
    {
        return UhdAdapterType::TREE_DATA;
    }

    size_t expectedFeatureCount() const override
    {
        return _numFeatures;
    }

    const std::string& getFeaturesHash() const override
    {
        return _featuresHash;
    }

    /// Get the number of trees in the ensemble.
    size_t treeCount() const
    {
        return _roots.size();
    }

    /// Get the model version string (RFC 0019 §13: model provenance).
    /// Empty if not set in the model.
    std::string getModelVersion() const override
    {
        return _modelVersion;
    }

    /// Get the list of GPU architectures the model was trained on.
    /// Empty if not set in the model.
    /// RFC 0019 §9.2: used for out-of-distribution detection.
    std::vector<std::string> getTrainingArches() const override
    {
        return _trainingArches;
    }

    /// Check if the given architecture was seen during training.
    /// Matches the bare target, ignoring runtime feature suffixes. An empty list is unrestricted.
    bool isTrainedForArch(const std::string& arch) const override;

private:
    struct Node
    {
        double value; // Threshold for a split, prediction for a leaf.
        std::array<uint32_t, 2> children; // Left, right; indices into _nodes.
        int32_t featureIndex; // -1 for a leaf; otherwise in [0, _numFeatures).
        bool defaultLeft;
        bool useLte;
    };

    TreeDataAdapter(std::vector<Node> nodes,
                    std::vector<uint32_t> roots,
                    std::string featuresHash,
                    size_t numFeatures,
                    double baseScore,
                    std::vector<std::string> trainingArches,
                    std::string modelVersion);

    static bool prepareTrees(const hipdnn_flatbuffers_sdk::data_objects::GbdtModel& model,
                             std::vector<Node>& nodes,
                             std::vector<uint32_t>& roots);

    template <bool CheckFeatureCount>
    double scorePrepared(const std::vector<double>& features) const;

    std::vector<Node> _nodes;
    std::vector<uint32_t> _roots;
    std::string _featuresHash;
    size_t _numFeatures;
    double _baseScore;

    // RFC 0019 §9.2, §13: Model provenance for out-of-distribution detection
    std::vector<std::string> _trainingArches;
    std::string _modelVersion;
};

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

inline std::unique_ptr<TreeDataAdapter>
    TreeDataAdapter::load(const std::string& modelPath,
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

inline std::unique_ptr<TreeDataAdapter>
    TreeDataAdapter::loadFromBuffer(const uint8_t* buffer,
                                    size_t size,
                                    const std::string& expectedFeaturesHash,
                                    const std::string& expectedModelHash)
{
    // Guard against null/empty buffer
    if(buffer == nullptr || size < sizeof(flatbuffers::uoffset_t) + 4 || size > 256 * 1024 * 1024)
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

    if(model->num_features() < 0)
    {
        HIPDNN_SDK_LOG_ERROR("TreeDataAdapter: negative feature count");
        return nullptr;
    }
    if(!std::isfinite(model->base_score()))
    {
        return nullptr;
    }

    std::vector<Node> nodes;
    std::vector<uint32_t> roots;
    if(!prepareTrees(*model, nodes, roots))
    {
        return nullptr;
    }

    const auto numFeatures = static_cast<size_t>(model->num_features());
    const double baseScore = model->base_score();

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

    return std::unique_ptr<TreeDataAdapter>(new TreeDataAdapter(std::move(nodes),
                                                                std::move(roots),
                                                                modelHash,
                                                                numFeatures,
                                                                baseScore,
                                                                std::move(trainingArches),
                                                                modelVersion));
}

inline TreeDataAdapter::TreeDataAdapter(std::vector<Node> nodes,
                                        std::vector<uint32_t> roots,
                                        std::string featuresHash,
                                        size_t numFeatures,
                                        double baseScore,
                                        std::vector<std::string> trainingArches,
                                        std::string modelVersion)
    : _nodes(std::move(nodes))
    , _roots(std::move(roots))
    , _featuresHash(std::move(featuresHash))
    , _numFeatures(numFeatures)
    , _baseScore(baseScore)
    , _trainingArches(std::move(trainingArches))
    , _modelVersion(std::move(modelVersion))
{
}

inline bool TreeDataAdapter::prepareTrees(const fb::GbdtModel& model,
                                          std::vector<Node>& nodes,
                                          std::vector<uint32_t>& roots)
{
    const auto* trees = model.trees();
    if(trees == nullptr)
    {
        return true;
    }

    const auto reject = [](size_t treeIndex, const char* reason) {
        HIPDNN_SDK_LOG_ERROR("TreeDataAdapter: malformed tree " << treeIndex << ": " << reason);
        return false;
    };

    // FlatBuffers verifies each vector, not the relationships between vectors.
    // Check their sizes before reserving or indexing the prepared representation.
    size_t totalNodes = 0;
    for(flatbuffers::uoffset_t t = 0; t < trees->size(); ++t)
    {
        const auto* tree = trees->Get(t);
        if(tree == nullptr || tree->left_children() == nullptr || tree->left_children()->empty()
           || tree->right_children() == nullptr || tree->feature_indices() == nullptr
           || tree->thresholds() == nullptr || tree->leaf_values() == nullptr)
        {
            return reject(t, "missing nodes or a required node array");
        }
        const auto count = tree->left_children()->size();
        if(tree->right_children()->size() != count || tree->feature_indices()->size() != count
           || tree->thresholds()->size() != count)
        {
            return reject(t, "node-parallel arrays have different lengths");
        }
        if(count > std::numeric_limits<uint32_t>::max() - totalNodes)
        {
            return reject(t, "ensemble exceeds the prepared node index range");
        }
        totalNodes += count;
    }
    nodes.reserve(totalNodes);
    roots.reserve(trees->size());

    // Kahn's algorithm checks all nodes, including branches not reached by a
    // particular feature row. Iterative validation also supports very deep trees.
    std::vector<uint32_t> incoming;
    std::vector<uint32_t> ready;
    for(flatbuffers::uoffset_t t = 0; t < trees->size(); ++t)
    {
        const auto* tree = trees->Get(t);
        const auto count = tree->left_children()->size();
        const auto offset = static_cast<uint32_t>(nodes.size());
        const auto* defaultLeft = tree->default_left();
        const auto* decisionLte = tree->decision_lte();
        const bool hasDecisionLte = decisionLte != nullptr && !decisionLte->empty();
        incoming.assign(count, 0);
        ready.clear();
        ready.reserve(count);
        roots.push_back(offset);

        for(flatbuffers::uoffset_t i = 0; i < count; ++i)
        {
            const int32_t left = tree->left_children()->Get(i);
            if(left == -1)
            {
                if(i >= tree->leaf_values()->size())
                {
                    return reject(t, "leaf has no prediction");
                }
                if(!std::isfinite(tree->leaf_values()->Get(i)))
                {
                    return reject(t, "leaf prediction is not finite");
                }
                nodes.push_back({tree->leaf_values()->Get(i), {0, 0}, -1, false, false});
                continue;
            }

            const int32_t right = tree->right_children()->Get(i);
            if(left < 0 || right < 0 || static_cast<uint32_t>(left) >= count
               || static_cast<uint32_t>(right) >= count)
            {
                return reject(t, "child index outside the tree");
            }
            if(!std::isfinite(tree->thresholds()->Get(i)))
            {
                return reject(t, "split threshold is not finite");
            }
            const int32_t feature = tree->feature_indices()->Get(i);
            if(feature < 0 || feature >= model.num_features())
            {
                return reject(t, "split feature outside the declared feature count");
            }
            const auto leftIndex = static_cast<uint32_t>(left);
            const auto rightIndex = static_cast<uint32_t>(right);
            ++incoming[leftIndex];
            ++incoming[rightIndex];
            nodes.push_back(
                {tree->thresholds()->Get(i),
                 {offset + leftIndex, offset + rightIndex},
                 feature,
                 defaultLeft != nullptr && i < defaultLeft->size() && defaultLeft->Get(i) != 0,
                 !hasDecisionLte || (i < decisionLte->size() && decisionLte->Get(i) != 0)});
        }

        for(uint32_t i = 0; i < count; ++i)
        {
            if(incoming[i] == 0)
            {
                ready.push_back(offset + i);
            }
        }
        for(size_t i = 0; i < ready.size(); ++i)
        {
            const auto& node = nodes[ready[i]];
            if(node.featureIndex < 0)
            {
                continue;
            }
            for(const auto child : node.children)
            {
                if(--incoming[child - offset] == 0)
                {
                    ready.push_back(child);
                }
            }
        }
        if(ready.size() != count)
        {
            return reject(t, "cycle in child indices");
        }
    }
    return true;
}

inline double TreeDataAdapter::score(const std::vector<double>& features) const
{
    if(_roots.empty())
    {
        return _baseScore;
    }
    // Validated splits are in range for a full-width row. Keep the existing
    // missing-feature behavior for short rows without burdening the usual path.
    return features.size() >= _numFeatures ? scorePrepared<false>(features)
                                           : scorePrepared<true>(features);
}

template <bool CheckFeatureCount>
inline double TreeDataAdapter::scorePrepared(const std::vector<double>& features) const
{
    double sum = 0.0;
    const auto* nodes = _nodes.data();
    for(const auto root : _roots)
    {
        const auto* node = nodes + root;
        while(node->featureIndex >= 0)
        {
            const auto feature = static_cast<size_t>(node->featureIndex);
            bool goLeft = node->defaultLeft;
            if(!CheckFeatureCount || feature < features.size())
            {
                const double value = features[feature];
                if(!std::isnan(value))
                {
                    goLeft = node->useLte ? value <= node->value : value < node->value;
                }
            }
            node = nodes + node->children[goLeft ? 0U : 1U];
        }
        sum += node->value;
    }

    // Preserve ensemble order and add the bias last. LightGBM's leaves already
    // include the learning rate; applying it again would double-count it.
    return _baseScore + sum;
}

inline bool TreeDataAdapter::isTrainedForArch(const std::string& arch) const
{
    // If no training arches specified, assume the model works for all arches
    if(_trainingArches.empty())
    {
        return true;
    }

    const auto target = stripArchFeatures(arch);
    return std::find(_trainingArches.begin(), _trainingArches.end(), target)
           != _trainingArches.end();
}

} // namespace hipdnn_plugin_sdk::uhd
