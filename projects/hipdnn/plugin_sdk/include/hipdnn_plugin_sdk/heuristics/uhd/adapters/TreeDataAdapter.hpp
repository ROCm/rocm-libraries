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

    /// Score a whole catalog at once, which is what a grouped model needs.
    ///
    /// A single-layer model answers per row, so the base implementation's loop is right for
    /// it and this override reduces to that. A grouped model cannot: picking the group is a
    /// decision across rows, and no per-row call can express it.
    std::vector<double> scoreBatch(const std::vector<std::vector<double>>& batch) const override;

    /// The slot `scoreBatch` groups on, so a ranker reports the same group the model used.
    int groupFeatureIndex() const override
    {
        return _groupFeatureIndex;
    }

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

    /// One group's layer-2 ensemble. Its roots index `_nodes` alongside layer 1's, so a
    /// grouped model is one allocation and one preparation path rather than a second kind
    /// of tree store that could validate differently from the first.
    struct Group
    {
        double value; // The grouping feature's value that selects this ensemble.
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
                    std::string modelVersion);

    /// Prepare one ensemble. Takes the tree vector rather than the model so that a group's
    /// trees go through exactly the validation layer 1's do -- a group whose trees were
    /// trusted where layer 1's were checked would be the one path into the walker that can
    /// index outside a row.
    static bool
        prepareTrees(const flatbuffers::Vector<
                         flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::GbdtTree>>* trees,
                     int32_t numFeatures,
                     std::vector<Node>& nodes,
                     std::vector<uint32_t>& roots);

    template <bool CheckFeatureCount>
    double scorePrepared(const std::vector<double>& features,
                         const std::vector<uint32_t>& roots) const;

    /// Sum one ensemble over a row, choosing the short-row path the same way `score` does.
    double scoreRoots(const std::vector<double>& features,
                      const std::vector<uint32_t>& roots) const
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
    //
    // ERROR, not WARN: RFC 0019 §12 requires "a clear error (not a warning) naming which of
    // the three checks failed and why, plus the fact that ranking degraded to static_order
    // and the estimate was reported as 0". The consequence is otherwise silent -- the engine
    // keeps answering and ranks by declared order -- so this line is the only trace. The
    // facts are named in the same order as every sibling check (which check, expected,
    // actual, consequence) so one grep finds them all.
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

    // Validate features hash (RFC 0019 §6.3 check 3, the signature the model was trained
    // against). ERROR for the reason given above.
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

    std::vector<Node> nodes;
    std::vector<uint32_t> roots;
    if(!prepareTrees(model->trees(), model->num_features(), nodes, roots))
    {
        return nullptr;
    }

    // A grouped model (RFC 0019 two-layer) decides twice: layer 1 picks the group, layer 2
    // orders within it. Both layers are prepared here so a malformed group is rejected at
    // load like a malformed ensemble, rather than at the first ranking that happens to
    // choose that group.
    std::vector<Group> groups;
    int groupFeatureIndex = -1;
    if(model->groups() != nullptr && !model->groups()->empty())
    {
        groupFeatureIndex = model->group_by_feature_index();
        // The walker reads the group from this slot of the feature row, so an index outside
        // the row is the same defect as a split feature outside it -- and unlike a split, it
        // would be read for every candidate of every ranking.
        if(groupFeatureIndex < 0 || groupFeatureIndex >= model->num_features())
        {
            HIPDNN_SDK_LOG_ERROR(
                "TreeDataAdapter: grouped model's group_by_feature_index "
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
            if(!prepareTrees(group->trees(), model->num_features(), nodes, groupRoots))
            {
                return nullptr;
            }
            groups.push_back({group->value(), std::move(groupRoots)});
        }
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
                                                                std::move(groups),
                                                                groupFeatureIndex,
                                                                modelHash,
                                                                numFeatures,
                                                                baseScore,
                                                                std::move(trainingArches),
                                                                modelVersion));
}

inline TreeDataAdapter::TreeDataAdapter(std::vector<Node> nodes,
                                        std::vector<uint32_t> roots,
                                        std::vector<Group> groups,
                                        int groupFeatureIndex,
                                        std::string featuresHash,
                                        size_t numFeatures,
                                        double baseScore,
                                        std::vector<std::string> trainingArches,
                                        std::string modelVersion)
    : _nodes(std::move(nodes))
    , _roots(std::move(roots))
    , _groups(std::move(groups))
    , _groupFeatureIndex(groupFeatureIndex)
    , _featuresHash(std::move(featuresHash))
    , _numFeatures(numFeatures)
    , _baseScore(baseScore)
    , _trainingArches(std::move(trainingArches))
    , _modelVersion(std::move(modelVersion))
{
}

inline bool TreeDataAdapter::prepareTrees(
    const flatbuffers::Vector<flatbuffers::Offset<fb::GbdtTree>>* trees,
    int32_t numFeatures,
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

    // FlatBuffers verifies each vector, not the relationships between vectors.
    // Check their sizes before reserving or indexing the prepared representation.
    // Counted from what `nodes` already holds, not from zero: a grouped model prepares every
    // layer into one store, and children are absolute indices into it, so the range that must
    // not overflow is the whole store rather than this ensemble's share of it.
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
        if(count > std::numeric_limits<uint32_t>::max() - totalNodes)
        {
            return reject(t, "ensemble exceeds the prepared node index range");
        }
        totalNodes += count;
    }
    nodes.reserve(totalNodes);
    roots.reserve(roots.size() + trees->size());

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
    // Validated splits are in range for a full-width row. Keep the existing
    // missing-feature behavior for short rows without burdening the usual path.
    return scoreRoots(features, _roots);
}

inline std::vector<double>
    TreeDataAdapter::scoreBatch(const std::vector<std::vector<double>>& batch) const
{
    if(_groupFeatureIndex < 0 || _groups.empty())
    {
        return IUhdAdapter::scoreBatch(batch);
    }

    // Layer 1 ranks the groups. Its score for a row stands for the group that row belongs
    // to, so the group's standing is the best its members achieve -- the same "achievable
    // when tuned" quantity layer 1 was trained on.
    const auto slot = static_cast<size_t>(_groupFeatureIndex);
    double bestGroupScore = -std::numeric_limits<double>::infinity();
    double chosenGroup = 0.0;
    bool chosen = false;
    for(const auto& row : batch)
    {
        if(slot >= row.size())
        {
            continue;
        }
        const double groupScore = score(row);
        if(!chosen || groupScore > bestGroupScore)
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

    // Layer 2 ranks within the chosen group. Everything outside it keeps -infinity, which
    // rankScored already treats as unusable, so a rejected group sorts last without needing
    // a new concept -- and cannot be picked by a tie.
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
        // A group layer 1 picked but layer 2 does not describe: rank it by layer 1 rather
        // than discarding it, so a partially trained artifact degrades instead of refusing
        // the only group it chose.
        scores[i] = within == nullptr || within->roots.empty() ? score(batch[i])
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
