// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "IUhdAdapter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/gbdt_model_generated.h>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/ScoreTransform.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/Sha256.hpp>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace hipdnn_plugin_sdk::uhd
{

/// @brief GBDT tree walker adapter; validates and flattens the FlatBuffer model at load.
class TreeDataAdapter : public IUhdAdapter
{
public:
    /// Load a GBDT model from a FlatBuffer file.
    /// @param expectedModelHash Optional SHA-256 of the model file; empty skips the check.
    /// @param objective The UHD's `objective` (`min` for a cost model).
    /// @param transform The UHD's `score.transform`; @p metric its `score.metric`.
    ///        objective, transform and metric are used only by a grouped model's group choice.
    /// @returns nullptr if loading or validation fails.
    static std::unique_ptr<TreeDataAdapter> load(const std::string& modelPath,
                                                 const std::string& expectedFeaturesHash,
                                                 const std::string& expectedModelHash = "",
                                                 const std::string& objective = "max",
                                                 const std::string& transform = "",
                                                 const std::string& metric = "");

    /// Load from an in-memory buffer, used only during this call; parameters as for load().
    /// @returns nullptr if validation fails.
    static std::unique_ptr<TreeDataAdapter> loadFromBuffer(const uint8_t* buffer,
                                                           size_t size,
                                                           const std::string& expectedFeaturesHash,
                                                           const std::string& expectedModelHash
                                                           = "",
                                                           const std::string& objective = "max",
                                                           const std::string& transform = "",
                                                           const std::string& metric = "");

    ~TreeDataAdapter() override = default;

    TreeDataAdapter(const TreeDataAdapter&) = delete;
    TreeDataAdapter& operator=(const TreeDataAdapter&) = delete;

    double score(const std::vector<double>& features) const override;

    /// Score a whole catalog. A grouped model needs this because picking the group is a
    /// decision across rows. Scores are raw, like score(); only the group choice reads the
    /// objective.
    std::vector<double> scoreBatch(const std::vector<std::vector<double>>& batch) const override;

    /// The slot `scoreBatch` groups on, so a ranker reports the same group the model used.
    int groupFeatureIndex() const override
    {
        return _groupFeatureIndex;
    }

    size_t expectedFeatureCount() const override
    {
        return _numFeatures;
    }

    const std::string& getFeaturesHash() const override
    {
        return _featuresHash;
    }

    /// Number of trees in the ensemble.
    size_t treeCount() const
    {
        return _roots.size();
    }

    /// Whether @p arch (bare target, feature suffixes ignored) was seen in training
    /// (RFC 0019 §8.3). An empty list is unrestricted.
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

    /// One group's layer-2 ensemble; its roots index the shared `_nodes` store.
    struct Group
    {
        double value; // Grouping-feature value that selects this ensemble.
        std::vector<uint32_t> roots;
    };

    TreeDataAdapter(std::vector<Node> nodes,
                    std::vector<uint32_t> roots,
                    std::vector<Group> groups,
                    int groupFeatureIndex,
                    std::string featuresHash,
                    size_t numFeatures,
                    double baseScore,
                    std::vector<std::string> trainingArches,
                    bool lowerIsBetter,
                    std::string transform,
                    bool positiveRequired);

    /// Prepare one ensemble. Shared by layer 1 and every group so all trees get the same
    /// bounds validation before the walker indexes rows with them. @p maxNodes caps the
    /// shared store across all calls.
    static bool prepareTrees(
        const flatbuffers::Vector<
            flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::GbdtTree>>* trees,
        int32_t numFeatures,
        size_t maxNodes,
        std::vector<Node>& nodes,
        std::vector<uint32_t>& roots);

    template <bool CheckFeatureCount>
    double scorePrepared(const std::vector<double>& features,
                         const std::vector<uint32_t>& roots) const;

    /// Sum one ensemble over a row, using the same short-row path as `score`.
    double scoreRoots(const std::vector<double>& features, const std::vector<uint32_t>& roots) const
    {
        if(roots.empty())
        {
            return _baseScore;
        }
        return features.size() >= _numFeatures ? scorePrepared<false>(features, roots)
                                               : scorePrepared<true>(features, roots);
    }

    std::vector<Node> _nodes;
    std::vector<uint32_t> _roots;
    std::vector<Group> _groups;
    int _groupFeatureIndex;
    std::string _featuresHash;
    size_t _numFeatures;
    double _baseScore;

    /// From the UHD's `objective`; the artifact does not record it, and the group choice
    /// needs it.
    bool _lowerIsBetter;

    /// The UHD's `score.transform`, and whether RFC 0019 §8.3 requires a positive recovered
    /// score; layer 1 applies the same rankability rule as layer 2.
    std::string _transform;
    bool _positiveRequired;

    // RFC 0019 §8.3: training arches for out-of-distribution detection.
    std::vector<std::string> _trainingArches;
};

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

inline std::unique_ptr<TreeDataAdapter>
    TreeDataAdapter::load(const std::string& modelPath,
                          const std::string& expectedFeaturesHash,
                          const std::string& expectedModelHash,
                          const std::string& objective,
                          const std::string& transform,
                          const std::string& metric)
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

    return loadFromBuffer(buffer.data(),
                          buffer.size(),
                          expectedFeaturesHash,
                          expectedModelHash,
                          objective,
                          transform,
                          metric);
}

inline std::unique_ptr<TreeDataAdapter>
    TreeDataAdapter::loadFromBuffer(const uint8_t* buffer,
                                    size_t size,
                                    const std::string& expectedFeaturesHash,
                                    const std::string& expectedModelHash,
                                    const std::string& objective,
                                    const std::string& transform,
                                    const std::string& metric)
{
    if(buffer == nullptr || size < sizeof(flatbuffers::uoffset_t) + 4
       || size > size_t{256} * 1024 * 1024)
    {
        return nullptr;
    }

    // RFC 0019 §9.2 integrity check. ERROR, not WARN (§12): this log is the only trace of the
    // fallback. Wording matches the sibling checks so one search finds them all.
    if(!expectedModelHash.empty())
    {
        const std::string actualHash = sha256(buffer, size);
        if(actualHash != expectedModelHash)
        {
            HIPDNN_SDK_LOG_ERROR(
                "TreeDataAdapter: model hash mismatch - expected='"
                << expectedModelHash << "' actual='" << actualHash
                << "'; the model is not used -- ranking degrades to static_order and an "
                   "engine estimate is reported as 0");
            return nullptr;
        }
    }

    if(!flatbuffers::BufferHasIdentifier(buffer, fb::GbdtModelIdentifier()))
    {
        return nullptr;
    }

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

    // RFC 0019 §6.3 check 3. ERROR for the reason given above.
    const std::string modelHash
        = model->features_hash() != nullptr ? model->features_hash()->str() : "";
    if(!expectedFeaturesHash.empty() && modelHash != expectedFeaturesHash)
    {
        HIPDNN_SDK_LOG_ERROR("TreeDataAdapter: features hash mismatch - expected='"
                             << expectedFeaturesHash << "' actual='" << modelHash
                             << "'; the model is not used -- ranking degrades to static_order "
                                "and an engine estimate is reported as 0");
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

    // Many `trees` entries may share one table, so the node count the offsets claim is not
    // bounded by the buffer. Each distinct node needs an int32 in left_children; more nodes
    // than that are repeats, and reserving for them could exhaust memory before validation.
    const size_t maxNodes
        = std::min<size_t>(size / sizeof(int32_t), std::numeric_limits<uint32_t>::max());
    std::vector<Node> nodes;
    std::vector<uint32_t> roots;
    if(!prepareTrees(model->trees(), model->num_features(), maxNodes, nodes, roots))
    {
        return nullptr;
    }

    // Two-layer model: layer 1 picks the group, layer 2 orders within it. Prepare both now so
    // a malformed group fails at load, not at the first ranking that picks it.
    std::vector<Group> groups;
    int groupFeatureIndex = -1;
    if(model->groups() != nullptr && !model->groups()->empty())
    {
        groupFeatureIndex = model->group_by_feature_index();
        // Read for every candidate, so it must lie inside the row like a split feature.
        if(groupFeatureIndex < 0 || groupFeatureIndex >= model->num_features())
        {
            HIPDNN_SDK_LOG_ERROR("TreeDataAdapter: grouped model's group_by_feature_index "
                                 << groupFeatureIndex << " is outside the declared feature count "
                                 << model->num_features());
            return nullptr;
        }
        groups.reserve(model->groups()->size());
        for(const auto* group : *model->groups())
        {
            if(group == nullptr)
            {
                HIPDNN_SDK_LOG_ERROR("TreeDataAdapter: null group");
                return nullptr;
            }
            std::vector<uint32_t> groupRoots;
            if(!prepareTrees(group->trees(), model->num_features(), maxNodes, nodes, groupRoots))
            {
                return nullptr;
            }
            groups.push_back({group->value(), std::move(groupRoots)});
        }
    }

    const auto numFeatures = static_cast<size_t>(model->num_features());
    const double baseScore = model->base_score();

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

    return std::unique_ptr<TreeDataAdapter>(
        new TreeDataAdapter(std::move(nodes),
                            std::move(roots),
                            std::move(groups),
                            groupFeatureIndex,
                            modelHash,
                            numFeatures,
                            baseScore,
                            std::move(trainingArches),
                            objective == "min",
                            transform,
                            score_transform::isPhysicalScore(metric, transform)));
}

inline TreeDataAdapter::TreeDataAdapter(std::vector<Node> nodes,
                                        std::vector<uint32_t> roots,
                                        std::vector<Group> groups,
                                        int groupFeatureIndex,
                                        std::string featuresHash,
                                        size_t numFeatures,
                                        double baseScore,
                                        std::vector<std::string> trainingArches,
                                        bool lowerIsBetter,
                                        std::string transform,
                                        bool positiveRequired)
    : _nodes(std::move(nodes))
    , _roots(std::move(roots))
    , _groups(std::move(groups))
    , _groupFeatureIndex(groupFeatureIndex)
    , _featuresHash(std::move(featuresHash))
    , _numFeatures(numFeatures)
    , _baseScore(baseScore)
    , _lowerIsBetter(lowerIsBetter)
    , _transform(std::move(transform))
    , _positiveRequired(positiveRequired)
    , _trainingArches(std::move(trainingArches))
{
}

inline bool TreeDataAdapter::prepareTrees(
    const flatbuffers::Vector<flatbuffers::Offset<fb::GbdtTree>>* trees,
    int32_t numFeatures,
    size_t maxNodes,
    std::vector<Node>& nodes,
    std::vector<uint32_t>& roots)
{
    if(trees == nullptr)
    {
        return true;
    }

    const auto reject = [](size_t treeIndex, const char* reason) {
        HIPDNN_SDK_LOG_ERROR("TreeDataAdapter: malformed tree " << treeIndex << ": " << reason);
        return false;
    };

    // FlatBuffers verifies each vector, not the relationships between vectors, so check sizes
    // before reserving or indexing. Count from nodes.size(): all layers share one store and
    // children are absolute indices into it. maxNodes fits uint32_t, so offsets stay exact.
    size_t totalNodes = nodes.size();
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
        if(count > maxNodes - totalNodes)
        {
            return reject(t, "ensemble has more nodes than the model buffer can hold");
        }
        totalNodes += count;
    }
    nodes.reserve(totalNodes);
    roots.reserve(roots.size() + trees->size());

    // Kahn's algorithm checks every node, including branches no row reaches, and is
    // iterative so very deep trees are fine.
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
            if(feature < 0 || feature >= numFeatures)
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
    // Validated splits are in range for a full-width row; short rows take the checked path.
    return scoreRoots(features, _roots);
}

inline std::vector<double>
    TreeDataAdapter::scoreBatch(const std::vector<std::vector<double>>& batch) const
{
    if(_groupFeatureIndex < 0 || _groups.empty())
    {
        return IUhdAdapter::scoreBatch(batch);
    }

    // Layer 1: a group's standing is the best layer-1 score among its rows, best in the
    // objective's direction. Raw scores are compared; every transform is increasing.
    // Only scores RFC 0019 §8.3 admits count. Ties go to the smaller group value so the
    // choice does not depend on row order.
    const auto slot = static_cast<size_t>(_groupFeatureIndex);
    double bestGroupScore = 0.0;
    double chosenGroup = 0.0;
    bool chosen = false;
    for(const auto& row : batch)
    {
        // No group value: NaN matches no group and would make the tie-break order-dependent.
        if(slot >= row.size() || std::isnan(row[slot]))
        {
            continue;
        }
        const double groupScore = score(row);
        if(!score_transform::isRankableScore(score_transform::applyInverse(groupScore, _transform),
                                             _positiveRequired))
        {
            continue;
        }
        const bool better
            = _lowerIsBetter ? groupScore < bestGroupScore : groupScore > bestGroupScore;
        if(!chosen || better || (groupScore == bestGroupScore && row[slot] < chosenGroup))
        {
            bestGroupScore = groupScore;
            chosenGroup = row[slot];
            chosen = true;
        }
    }

    std::vector<double> scores(batch.size(), -std::numeric_limits<double>::infinity());
    if(!chosen)
    {
        return scores;
    }

    // Layer 2 ranks within the chosen group; all other rows stay -infinity (unusable).
    const Group* within = nullptr;
    for(const auto& candidate : _groups)
    {
        if(candidate.value == chosenGroup)
        {
            within = &candidate;
            break;
        }
    }

    for(size_t i = 0; i < batch.size(); ++i)
    {
        if(slot >= batch[i].size() || batch[i][slot] != chosenGroup)
        {
            continue;
        }
        // Chosen group missing from layer 2: fall back to layer 1 rather than reject it.
        scores[i] = within == nullptr || within->roots.empty()
                        ? score(batch[i])
                        : scoreRoots(batch[i], within->roots);
    }
    return scores;
}

template <bool CheckFeatureCount>
inline double TreeDataAdapter::scorePrepared(const std::vector<double>& features,
                                             const std::vector<uint32_t>& roots) const
{
    double sum = 0.0;
    const auto* nodes = _nodes.data();
    for(const auto root : roots)
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

    // Add the bias last. LightGBM leaves already include the learning rate; don't reapply.
    return _baseScore + sum;
}

inline bool TreeDataAdapter::isTrainedForArch(const std::string& arch) const
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
