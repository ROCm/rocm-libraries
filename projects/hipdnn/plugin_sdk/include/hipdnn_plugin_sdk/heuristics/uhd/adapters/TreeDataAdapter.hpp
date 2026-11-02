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
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hipdnn_plugin_sdk::uhd
{

namespace detail
{

namespace fbs = hipdnn_flatbuffers_sdk::data_objects;

/// Calls @p f with @p table as the IntegerArray member @p type names. False, calling
/// nothing, for NONE or a tag this runtime does not know: the one list of members it reads.
template <typename F>
inline bool visitArray(fbs::IntegerArray type, const void* table, F&& f)
{
    switch(type)
    {
    case fbs::IntegerArray::Int8Array:
        f(static_cast<const fbs::Int8Array*>(table));
        return true;
    case fbs::IntegerArray::UInt8Array:
        f(static_cast<const fbs::UInt8Array*>(table));
        return true;
    case fbs::IntegerArray::Int16Array:
        f(static_cast<const fbs::Int16Array*>(table));
        return true;
    case fbs::IntegerArray::UInt16Array:
        f(static_cast<const fbs::UInt16Array*>(table));
        return true;
    case fbs::IntegerArray::Int32Array:
        f(static_cast<const fbs::Int32Array*>(table));
        return true;
    default:
        return false;
    }
}

/// As above, for a RealArray.
template <typename F>
inline bool visitArray(fbs::RealArray type, const void* table, F&& f)
{
    switch(type)
    {
    case fbs::RealArray::Float32Array:
        f(static_cast<const fbs::Float32Array*>(table));
        return true;
    case fbs::RealArray::Float64Array:
        f(static_cast<const fbs::Float64Array*>(table));
        return true;
    default:
        return false;
    }
}

/// Bytes one element of the typed array @p Table occupies.
template <typename Table>
constexpr size_t elementBytesOf()
{
    using Values = std::remove_pointer_t<decltype(std::declval<const Table&>().values())>;
    return sizeof(typename Values::return_type);
}

/// The union a tag belongs to, as a refusal names it.
inline const char* kindOf(fbs::IntegerArray /*type*/)
{
    return "integer";
}

inline const char* kindOf(fbs::RealArray /*type*/)
{
    return "real";
}

/// A typed array's values, in whichever type its tag names, widened to @p T. False when the
/// member is absent, unknown to this runtime or holds no values, with @p why saying which:
/// "is absent", "is stored as <kind> array type N, which this runtime cannot read", or
/// "has no values".
template <typename Union, typename T>
inline bool decode(Union type, const void* table, std::vector<T>& out, std::string& why)
{
    bool present = false;
    const bool known = visitArray(type, table, [&](const auto* typed) {
        present = typed != nullptr && typed->values() != nullptr;
        if(present)
        {
            out.assign(typed->values()->begin(), typed->values()->end());
        }
    });
    if(type == Union::NONE)
    {
        why = "is absent";
    }
    else if(!known)
    {
        why = std::string("is stored as ") + kindOf(type) + " array type "
              + std::to_string(static_cast<int>(type)) + ", which this runtime cannot read";
    }
    else if(!present)
    {
        why = "has no values";
    }
    return known && present;
}

/// How many values a RealArray holds, in whichever type its tag names; 0 when unknown.
inline size_t length(fbs::RealArray type, const void* table)
{
    size_t count = 0;
    visitArray(type, table, [&](const auto* typed) {
        count = typed == nullptr || typed->values() == nullptr ? 0 : typed->values()->size();
    });
    return count;
}

/// Bytes one element of an IntegerArray / RealArray of this tag occupies; 0 when unknown.
template <typename Union>
inline size_t elementBytes(Union type)
{
    size_t bytes = 0;
    visitArray(type, nullptr, [&](const auto* typed) {
        bytes = elementBytesOf<std::remove_cv_t<std::remove_pointer_t<decltype(typed)>>>();
    });
    return bytes;
}

} // namespace detail

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
    /// bounds validation before the walker indexes rows with them. @p maxBytes caps, across
    /// all calls, the bytes the nodes' arrays claim (`budget`): trees may share arrays, so
    /// the node count is bounded by the buffer, not by the offsets. Both encodings decode
    /// into the one node store the walker reads; @p thresholdTable serves split/leaf trees.
    static bool prepareTrees(
        const flatbuffers::Vector<
            flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::GbdtTree>>* trees,
        int32_t numFeatures,
        size_t maxBytes,
        const std::vector<double>& thresholdTable,
        size_t& budget,
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

    // Split/leaf-encoded trees index one table of distinct thresholds, stored in whichever
    // real type the model declares for it.
    std::vector<double> thresholdTable;
    if(model->threshold_table_type() != fb::RealArray::NONE)
    {
        std::string why;
        if(!detail::decode(
               model->threshold_table_type(), model->threshold_table(), thresholdTable, why))
        {
            HIPDNN_SDK_LOG_ERROR("TreeDataAdapter: threshold_table " << why);
            return nullptr;
        }
    }

    // Many `trees` entries may share one array, so the node count the offsets claim is not
    // bounded by the buffer. Each distinct node needs its child entry (an int32 per node
    // array node; a split's child width or a leaf's value width otherwise); claiming more
    // bytes than the buffer holds means repeats, and reserving for them could exhaust memory
    // before validation.
    const size_t maxBytes
        = std::min<size_t>(size, size_t{std::numeric_limits<uint32_t>::max()} * sizeof(int32_t));
    size_t budget = 0;
    std::vector<Node> nodes;
    std::vector<uint32_t> roots;
    if(!prepareTrees(
           model->trees(), model->num_features(), maxBytes, thresholdTable, budget, nodes, roots))
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
        std::set<double> groupValues;
        for(const auto* group : *model->groups())
        {
            if(group == nullptr || !std::isfinite(group->value())
               || !groupValues.insert(group->value()).second)
            {
                HIPDNN_SDK_LOG_ERROR("TreeDataAdapter: group values must be finite and unique");
                return nullptr;
            }
            std::vector<uint32_t> groupRoots;
            if(!prepareTrees(group->trees(),
                             model->num_features(),
                             maxBytes,
                             thresholdTable,
                             budget,
                             nodes,
                             groupRoots))
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
    size_t maxBytes,
    const std::vector<double>& thresholdTable,
    size_t& budget,
    std::vector<Node>& nodes,
    std::vector<uint32_t>& roots)
{
    if(trees == nullptr)
    {
        return true;
    }

    const auto reject = [](size_t treeIndex, const std::string& reason) {
        HIPDNN_SDK_LOG_ERROR("TreeDataAdapter: malformed tree " << treeIndex << ": " << reason);
        return false;
    };

    // FlatBuffers verifies each vector, not the relationships between vectors, so check sizes
    // before reserving or indexing. Count from nodes.size(): all layers share one store and
    // children are absolute indices into it.
    size_t totalNodes = nodes.size();
    for(flatbuffers::uoffset_t t = 0; t < trees->size(); ++t)
    {
        const auto* tree = trees->Get(t);
        if(tree == nullptr)
        {
            return reject(t, "missing nodes or a required node array");
        }
        size_t count = 0;
        size_t bytes = 0;
        if(tree->leaves_type() != fb::RealArray::NONE)
        {
            const size_t leafBytes = detail::elementBytes(tree->leaves_type());
            const size_t childBytes = detail::elementBytes(tree->split_left_type());
            if(leafBytes == 0)
            {
                return reject(t, "leaves are stored in a type this runtime cannot read");
            }
            const size_t leafCount = detail::length(tree->leaves_type(), tree->leaves());
            // A tree has one leaf more than it has splits; the split arrays are checked
            // against this when the tree is decoded below.
            const size_t splitCount = leafCount == 0 ? 0 : leafCount - 1;
            if(splitCount > 0 && childBytes == 0)
            {
                return reject(t, "split children are stored in a type this runtime cannot read");
            }
            count = leafCount;
            count += splitCount;
            bytes = leafCount * leafBytes + splitCount * childBytes;
        }
        else
        {
            if(tree->left_children() == nullptr || tree->left_children()->empty())
            {
                return reject(t, "missing nodes or a required node array");
            }
            count = tree->left_children()->size();
            bytes = count * sizeof(int32_t);
        }
        if(count == 0)
        {
            return reject(t, "missing nodes or a required node array");
        }
        if(bytes > maxBytes - budget)
        {
            return reject(t, "ensemble has more nodes than the model buffer can hold");
        }
        budget += bytes;
        totalNodes += count;
    }
    nodes.reserve(totalNodes);
    roots.reserve(roots.size() + trees->size());

    // Kahn's algorithm checks every node, including branches no row reaches, and is
    // iterative so very deep trees are fine.
    std::vector<uint32_t> incoming;
    std::vector<uint32_t> ready;
    std::vector<int32_t> features;
    std::vector<int32_t> thresholdIds;
    std::vector<int32_t> lefts;
    std::vector<int32_t> rights;
    std::vector<double> leaves;
    for(flatbuffers::uoffset_t t = 0; t < trees->size(); ++t)
    {
        const auto* tree = trees->Get(t);
        const auto offset = static_cast<uint32_t>(nodes.size());
        roots.push_back(offset);
        uint32_t count = 0;

        // Either encoding's splits and leaves, once their children are in range: the same
        // checks, in the same order, and the same nodes. Returns the refusal, or nullptr.
        const auto addSplit = [&](double threshold,
                                  int32_t feature,
                                  uint32_t left,
                                  uint32_t right,
                                  bool defaultLeft,
                                  bool lte) -> const char* {
            if(!std::isfinite(threshold))
            {
                return "split threshold is not finite";
            }
            if(feature < 0 || feature >= numFeatures)
            {
                return "split feature outside the declared feature count";
            }
            ++incoming[left];
            ++incoming[right];
            nodes.push_back(
                {threshold, {offset + left, offset + right}, feature, defaultLeft, lte});
            return nullptr;
        };
        const auto addLeaf = [&](double value) -> const char* {
            if(!std::isfinite(value))
            {
                return "leaf prediction is not finite";
            }
            nodes.push_back({value, {0, 0}, -1, false, false});
            return nullptr;
        };

        if(tree->leaves_type() != fb::RealArray::NONE)
        {
            // Split/leaf encoding: splits at offset + i, then leaves at offset + splits + j.
            std::string why;
            if(!detail::decode(tree->leaves_type(), tree->leaves(), leaves, why) || leaves.empty())
            {
                return reject(t, "leaves " + (why.empty() ? std::string("are empty") : why));
            }
            const auto splits = static_cast<uint32_t>(leaves.size() - 1);
            const auto leafCount = static_cast<uint32_t>(leaves.size());
            count = splits + leafCount;
            const auto integers = [&](const char* name,
                                      fb::IntegerArray type,
                                      const void* table,
                                      std::vector<int32_t>& out) {
                std::string reason;
                return detail::decode(type, table, out, reason)
                       || reject(t, std::string(name) + " " + reason);
            };
            if(splits > 0)
            {
                if(!integers("split_features",
                             tree->split_features_type(),
                             tree->split_features(),
                             features)
                   || !integers("split_thresholds",
                                tree->split_thresholds_type(),
                                tree->split_thresholds(),
                                thresholdIds)
                   || !integers("split_left", tree->split_left_type(), tree->split_left(), lefts)
                   || !integers(
                       "split_right", tree->split_right_type(), tree->split_right(), rights))
                {
                    return false;
                }
            }
            else if(tree->split_features_type() != fb::IntegerArray::NONE
                    || tree->split_thresholds_type() != fb::IntegerArray::NONE
                    || tree->split_left_type() != fb::IntegerArray::NONE
                    || tree->split_right_type() != fb::IntegerArray::NONE)
            {
                return reject(t, "a single-leaf tree has split arrays");
            }
            const auto* defaultLeft = tree->split_default_left();
            const auto* decisionLte = tree->split_decision_lte();
            if((splits > 0
                && (features.size() != splits || thresholdIds.size() != splits
                    || lefts.size() != splits || rights.size() != splits))
               || (defaultLeft != nullptr && defaultLeft->size() != splits)
               || (decisionLte != nullptr && decisionLte->size() != splits))
            {
                return reject(t,
                              "split arrays do not hold one entry per split (one fewer "
                              "than the leaves)");
            }

            // Child c >= 0 is split c; c < 0 is leaf ~c.
            const auto node = [&](int32_t child, uint32_t& at) {
                if(child >= 0)
                {
                    at = static_cast<uint32_t>(child);
                    return at < splits;
                }
                at = splits + static_cast<uint32_t>(~child);
                return static_cast<uint32_t>(~child) < leafCount;
            };
            incoming.assign(count, 0);
            for(uint32_t i = 0; i < splits; ++i)
            {
                uint32_t left = 0;
                uint32_t right = 0;
                if(!node(lefts[i], left) || !node(rights[i], right))
                {
                    return reject(t, "child index outside the tree");
                }
                const int32_t id = thresholdIds[i];
                if(id < 0 || static_cast<size_t>(id) >= thresholdTable.size())
                {
                    return reject(t, "split threshold index outside the threshold table");
                }
                if(const char* refusal
                   = addSplit(thresholdTable[static_cast<size_t>(id)],
                              features[i],
                              left,
                              right,
                              defaultLeft != nullptr ? defaultLeft->Get(i) != 0
                                                     : tree->default_left_fill(),
                              decisionLte == nullptr || decisionLte->Get(i) != 0))
                {
                    return reject(t, refusal);
                }
            }
            for(const double leaf : leaves)
            {
                if(const char* refusal = addLeaf(leaf))
                {
                    return reject(t, refusal);
                }
            }
        }
        else
        {
            if(tree->right_children() == nullptr || tree->feature_indices() == nullptr
               || tree->thresholds() == nullptr || tree->leaf_values() == nullptr)
            {
                return reject(t, "missing nodes or a required node array");
            }
            count = tree->left_children()->size();
            if(tree->right_children()->size() != count || tree->feature_indices()->size() != count
               || tree->thresholds()->size() != count)
            {
                return reject(t, "node-parallel arrays have different lengths");
            }
            const auto* defaultLeft = tree->default_left();
            const auto* decisionLte = tree->decision_lte();
            const bool hasDecisionLte = decisionLte != nullptr && !decisionLte->empty();
            incoming.assign(count, 0);

            for(flatbuffers::uoffset_t i = 0; i < count; ++i)
            {
                const int32_t left = tree->left_children()->Get(i);
                if(left == -1)
                {
                    if(i >= tree->leaf_values()->size())
                    {
                        return reject(t, "leaf has no prediction");
                    }
                    if(const char* refusal = addLeaf(tree->leaf_values()->Get(i)))
                    {
                        return reject(t, refusal);
                    }
                    continue;
                }

                const int32_t right = tree->right_children()->Get(i);
                if(left < 0 || right < 0 || static_cast<uint32_t>(left) >= count
                   || static_cast<uint32_t>(right) >= count)
                {
                    return reject(t, "child index outside the tree");
                }
                if(const char* refusal = addSplit(
                       tree->thresholds()->Get(i),
                       tree->feature_indices()->Get(i),
                       static_cast<uint32_t>(left),
                       static_cast<uint32_t>(right),
                       defaultLeft != nullptr && i < defaultLeft->size()
                           && defaultLeft->Get(i) != 0,
                       !hasDecisionLte || (i < decisionLte->size() && decisionLte->Get(i) != 0)))
                {
                    return reject(t, refusal);
                }
            }
        }

        ready.clear();
        ready.reserve(count);
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
        // Nonfinite group values cannot name an admitted group.
        if(slot >= row.size() || !std::isfinite(row[slot]))
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
