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
#include <numeric>
#include <optional>
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
///
/// Splits compare ranks, not values. At load every feature's distinct split thresholds are
/// sorted into a table and each split stores its threshold's position in it. A row is mapped
/// once to positions (one binary search per feature the model reads), and every split then
/// compares two integers: `x <= t[r]` exactly when fewer than r + 1 thresholds lie below x,
/// and `x < t[r]` exactly when fewer than r + 1 lie at or below it. Decisions, leaf values
/// and the order leaves are summed in are those of a direct walk, so scores are identical.
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
    /// decision across rows, and an early-exit schedule because pruning is. Scores are raw,
    /// like score(); only the group choice and the pruning read the objective.
    BatchScores scoreBatch(const FeatureRows& rows) const override;

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

    /// Number of trees in the ensemble (layer 1 for a grouped model).
    size_t treeCount() const
    {
        return _roots.size();
    }

    /// One stage of the model's early-exit schedule: after `trees` trees, keep the `keep`
    /// best candidates.
    struct EarlyExitStage
    {
        uint32_t trees;
        uint32_t keep;
    };

    /// The early-exit schedule scoreBatch applies; empty when every candidate is fully
    /// scored (no schedule, or a grouped model, whose schedule is ignored).
    const std::vector<EarlyExitStage>& earlyExit() const
    {
        return _earlyExit;
    }

    /// Whether @p arch (bare target, feature suffixes ignored) was seen in training
    /// (RFC 0019 §8.3). An empty list is unrestricted.
    bool isTrainedForArch(const std::string& arch) const override;

private:
    /// A node as loading validates it; scoring uses the compiled form below.
    struct Node
    {
        double value; // Threshold for a split, prediction for a leaf.
        std::array<uint32_t, 2> children; // Left, right; indices into the node store.
        int32_t featureIndex; // -1 for a leaf; otherwise in [0, _numFeatures).
        bool defaultLeft;
        bool useLte;
    };

    /// One group's validated layer-2 roots, as loading prepares them.
    struct PreparedGroup
    {
        double value; // Grouping-feature value that selects this ensemble.
        std::vector<uint32_t> roots;
    };

    /// A compiled split. A child >= 0 is a split index; a child c < 0 is leaf ~c. `slot`
    /// indexes a quantized row and carries the missing-value direction in its top bit.
    struct Split
    {
        int32_t left;
        int32_t right;
        uint32_t rank; // Position of the split's threshold in its feature's sorted table.
        uint32_t slot;
    };

    /// One group's layer-2 ensemble, as compiled tree references.
    struct Group
    {
        double value;
        std::vector<int32_t> roots;
    };

    static constexpr uint32_t DEFAULT_LEFT_BIT = 0x80000000U;
    static constexpr uint32_t SLOT_MASK = 0x7FFFFFFFU;
    /// A quantized value no split compares with: the feature is missing (NaN or past the
    /// row's end) and the split takes its default direction.
    static constexpr uint32_t MISSING = 0xFFFFFFFFU;

    TreeDataAdapter(const std::vector<Node>& nodes,
                    const std::vector<uint32_t>& roots,
                    const std::vector<PreparedGroup>& groups,
                    std::vector<EarlyExitStage> earlyExit,
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
    /// into one node store; @p thresholdTable serves split/leaf trees.
    static bool prepareTrees(
        const flatbuffers::Vector<
            flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::GbdtTree>>* trees,
        int32_t numFeatures,
        size_t maxBytes,
        const std::vector<double>& thresholdTable,
        size_t& budget,
        std::vector<Node>& nodes,
        std::vector<uint32_t>& roots);

    /// The schedule's stages, refusing (nullopt) a malformed one with the reason logged.
    static std::optional<std::vector<EarlyExitStage>> readEarlyExit(
        const flatbuffers::Vector<
            flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::EarlyExitStage>>* stages,
        size_t treeCount);

    /// Values per quantized row: two per feature the model reads (`<=` and `<` positions).
    size_t quantizedWidth() const
    {
        return 2 * _features.size();
    }

    /// Map @p row (@p width values) to threshold positions in @p out. A feature whose value
    /// equals @p previous's (the row before, already mapped to @p previousOut) is copied
    /// rather than searched: a catalog's rows share every problem feature.
    void quantize(const double* row,
                  size_t width,
                  const double* previous,
                  const uint32_t* previousOut,
                  uint32_t* out) const;

    /// The leaf a quantized row reaches from tree reference @p ref.
    double leafOf(const uint32_t* quantized, int32_t ref) const;

    /// Add the leaves of @p roots [@p begin, @p end) to @p sum, in tree order.
    void accumulate(const uint32_t* quantized,
                    const std::vector<int32_t>& roots,
                    size_t begin,
                    size_t end,
                    double& sum) const
    {
        for(size_t t = begin; t < end; ++t)
        {
            sum += leafOf(quantized, roots[t]);
        }
    }

    /// base_score plus every leaf of @p roots; base_score alone for an empty ensemble.
    double scoreRoots(const uint32_t* quantized, const std::vector<int32_t>& roots) const
    {
        if(roots.empty())
        {
            return _baseScore;
        }
        double sum = 0.0;
        accumulate(quantized, roots, 0, roots.size(), sum);
        // Add the bias last. LightGBM leaves already include the learning rate; don't reapply.
        return _baseScore + sum;
    }

    BatchScores scoreGrouped(const FeatureRows& rows, const std::vector<uint32_t>& quantized) const;
    BatchScores scoreStaged(size_t count, const std::vector<uint32_t>& quantized) const;

    std::vector<Split> _splits;
    std::vector<double> _leaves;
    std::vector<int32_t> _roots;
    std::vector<Group> _groups;
    std::vector<EarlyExitStage> _earlyExit;

    /// Quantizer: per feature the model reads, its row index, whether any split on it uses
    /// `<`, and its sorted distinct thresholds `_thresholds[_tableOffsets[k], ...[k + 1])`.
    std::vector<uint32_t> _features;
    std::vector<uint8_t> _strict;
    std::vector<uint32_t> _tableOffsets;
    std::vector<double> _thresholds;

    int _groupFeatureIndex;
    std::string _featuresHash;
    size_t _numFeatures;
    double _baseScore;

    /// From the UHD's `objective`; the artifact does not record it, and the group choice
    /// and early-exit pruning need it.
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
    std::vector<PreparedGroup> groups;
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

    // An early-exit schedule prunes layer-1 rows. A grouped model's layer 1 ranks groups, not
    // candidates, so its schedule has nothing to prune and is ignored rather than refused.
    std::vector<EarlyExitStage> earlyExit;
    if(model->early_exit() != nullptr && !model->early_exit()->empty())
    {
        if(!groups.empty())
        {
            HIPDNN_SDK_LOG_WARN("TreeDataAdapter: early_exit is ignored on a grouped model; "
                                "every candidate is fully scored");
        }
        else
        {
            auto stages = readEarlyExit(model->early_exit(), roots.size());
            if(!stages)
            {
                return nullptr;
            }
            earlyExit = std::move(*stages);
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
        new TreeDataAdapter(nodes,
                            roots,
                            groups,
                            std::move(earlyExit),
                            groupFeatureIndex,
                            modelHash,
                            numFeatures,
                            baseScore,
                            std::move(trainingArches),
                            objective == "min",
                            transform,
                            score_transform::isPhysicalScore(metric, transform)));
}

inline std::optional<std::vector<TreeDataAdapter::EarlyExitStage>> TreeDataAdapter::readEarlyExit(
    const flatbuffers::Vector<flatbuffers::Offset<fb::EarlyExitStage>>* stages, size_t treeCount)
{
    const auto refuse = [](size_t stage, const char* reason) {
        HIPDNN_SDK_LOG_ERROR("TreeDataAdapter: early_exit stage " << stage << ": " << reason);
        return std::nullopt;
    };
    // scoreBatch reports each row's stage in a uint8_t.
    if(stages->size() > std::numeric_limits<uint8_t>::max())
    {
        return refuse(stages->size(), "the schedule has more than 255 stages");
    }
    std::vector<EarlyExitStage> schedule;
    schedule.reserve(stages->size());
    for(flatbuffers::uoffset_t i = 0; i < stages->size(); ++i)
    {
        const auto* stage = stages->Get(i);
        if(stage == nullptr)
        {
            return refuse(i, "the stage is missing");
        }
        if(stage->trees() == 0 || stage->trees() >= treeCount)
        {
            return refuse(i, "trees must be at least 1 and fewer than the model's trees");
        }
        if(stage->keep() == 0)
        {
            return refuse(i, "keep must be at least 1");
        }
        if(!schedule.empty()
           && (stage->trees() <= schedule.back().trees || stage->keep() >= schedule.back().keep))
        {
            return refuse(i, "stages must increase in trees and decrease in keep");
        }
        schedule.push_back({stage->trees(), stage->keep()});
    }
    return schedule;
}

inline TreeDataAdapter::TreeDataAdapter(const std::vector<Node>& nodes,
                                        const std::vector<uint32_t>& roots,
                                        const std::vector<PreparedGroup>& groups,
                                        std::vector<EarlyExitStage> earlyExit,
                                        int groupFeatureIndex,
                                        std::string featuresHash,
                                        size_t numFeatures,
                                        double baseScore,
                                        std::vector<std::string> trainingArches,
                                        bool lowerIsBetter,
                                        std::string transform,
                                        bool positiveRequired)
    : _earlyExit(std::move(earlyExit))
    , _groupFeatureIndex(groupFeatureIndex)
    , _featuresHash(std::move(featuresHash))
    , _numFeatures(numFeatures)
    , _baseScore(baseScore)
    , _lowerIsBetter(lowerIsBetter)
    , _transform(std::move(transform))
    , _positiveRequired(positiveRequired)
    , _trainingArches(std::move(trainingArches))
{
    // Number the splits and leaves in node order; a tree reference is a split index or the
    // complement of a leaf index.
    std::vector<int32_t> reference(nodes.size());
    int32_t splitCount = 0;
    int32_t leafCount = 0;
    for(size_t n = 0; n < nodes.size(); ++n)
    {
        reference[n] = nodes[n].featureIndex >= 0 ? splitCount++ : ~leafCount++;
    }

    // Each feature's sorted distinct thresholds, and whether any split on it uses `<`.
    // Built from (feature, threshold) pairs so the work follows the splits, not the declared
    // feature count.
    std::vector<std::pair<int32_t, double>> thresholds;
    thresholds.reserve(static_cast<size_t>(splitCount));
    std::vector<std::pair<int32_t, bool>> strict;
    for(const auto& node : nodes)
    {
        if(node.featureIndex >= 0)
        {
            thresholds.emplace_back(node.featureIndex, node.value);
            strict.emplace_back(node.featureIndex, !node.useLte);
        }
    }
    std::sort(thresholds.begin(), thresholds.end());
    thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());
    std::sort(strict.begin(), strict.end());
    for(const auto& [feature, threshold] : thresholds)
    {
        if(_features.empty() || _features.back() != static_cast<uint32_t>(feature))
        {
            _features.push_back(static_cast<uint32_t>(feature));
            _tableOffsets.push_back(static_cast<uint32_t>(_thresholds.size()));
            // `strict` is sorted with `true` last within a feature.
            const auto last = std::upper_bound(
                strict.begin(), strict.end(), std::pair<int32_t, bool>{feature, true});
            _strict.push_back(static_cast<uint8_t>(
                last != strict.begin() && (last - 1)->first == feature && (last - 1)->second));
        }
        _thresholds.push_back(threshold);
    }
    _tableOffsets.push_back(static_cast<uint32_t>(_thresholds.size()));

    _splits.reserve(static_cast<size_t>(splitCount));
    _leaves.reserve(static_cast<size_t>(leafCount));
    for(const auto& node : nodes)
    {
        if(node.featureIndex < 0)
        {
            _leaves.push_back(node.value);
            continue;
        }
        const auto feature = static_cast<uint32_t>(node.featureIndex);
        const auto k = static_cast<uint32_t>(
            std::lower_bound(_features.begin(), _features.end(), feature) - _features.begin());
        const double* table = _thresholds.data() + _tableOffsets[k];
        const double* tableEnd = _thresholds.data() + _tableOffsets[k + 1];
        const auto rank
            = static_cast<uint32_t>(std::lower_bound(table, tableEnd, node.value) - table);
        const uint32_t slot
            = (2 * k + (node.useLte ? 0U : 1U)) | (node.defaultLeft ? DEFAULT_LEFT_BIT : 0U);
        _splits.push_back({reference[node.children[0]], reference[node.children[1]], rank, slot});
    }

    _roots.reserve(roots.size());
    for(const auto root : roots)
    {
        _roots.push_back(reference[root]);
    }
    _groups.reserve(groups.size());
    for(const auto& group : groups)
    {
        Group compiled{group.value, {}};
        compiled.roots.reserve(group.roots.size());
        for(const auto root : group.roots)
        {
            compiled.roots.push_back(reference[root]);
        }
        _groups.push_back(std::move(compiled));
    }
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

inline void TreeDataAdapter::quantize(const double* row,
                                      size_t width,
                                      const double* previous,
                                      const uint32_t* previousOut,
                                      uint32_t* out) const
{
    for(size_t k = 0; k < _features.size(); ++k)
    {
        const size_t feature = _features[k];
        // A feature past the row's end reads as missing, as a NaN does.
        const double value
            = feature < width ? row[feature] : std::numeric_limits<double>::quiet_NaN();
        if(std::isnan(value))
        {
            out[2 * k] = MISSING;
            out[(2 * k) + 1] = MISSING;
            continue;
        }
        // Equal values (including -0.0 and 0.0, which every split also treats as equal) map
        // to equal positions.
        if(previous != nullptr && previous[feature] == value)
        {
            out[2 * k] = previousOut[2 * k];
            out[(2 * k) + 1] = previousOut[(2 * k) + 1];
            continue;
        }
        const double* table = _thresholds.data() + _tableOffsets[k];
        const double* tableEnd = _thresholds.data() + _tableOffsets[k + 1];
        // Thresholds below the value: `value <= t[r]` exactly when this is <= r.
        out[2 * k] = static_cast<uint32_t>(std::lower_bound(table, tableEnd, value) - table);
        // Thresholds at or below it: `value < t[r]` exactly when this is <= r.
        out[(2 * k) + 1]
            = _strict[k] != 0
                  ? static_cast<uint32_t>(std::upper_bound(table, tableEnd, value) - table)
                  : out[2 * k];
    }
}

inline double TreeDataAdapter::leafOf(const uint32_t* quantized, int32_t ref) const
{
    const Split* splits = _splits.data();
    while(ref >= 0)
    {
        const Split& split = splits[ref];
        const uint32_t position = quantized[split.slot & SLOT_MASK];
        const bool goLeft
            = position == MISSING ? (split.slot & DEFAULT_LEFT_BIT) != 0 : position <= split.rank;
        ref = goLeft ? split.left : split.right;
    }
    const int32_t leaf = ~ref;
    return _leaves[static_cast<size_t>(leaf)];
}

inline double TreeDataAdapter::score(const std::vector<double>& features) const
{
    std::vector<uint32_t> quantized(quantizedWidth());
    quantize(features.data(), features.size(), nullptr, nullptr, quantized.data());
    return scoreRoots(quantized.data(), _roots);
}

inline BatchScores TreeDataAdapter::scoreBatch(const FeatureRows& rows) const
{
    const size_t width = quantizedWidth();
    std::vector<uint32_t> quantized(rows.count * width);
    for(size_t i = 0; i < rows.count; ++i)
    {
        quantize(rows.row(i),
                 rows.width,
                 i == 0 ? nullptr : rows.row(i - 1),
                 i == 0 ? nullptr : quantized.data() + ((i - 1) * width),
                 quantized.data() + (i * width));
    }

    if(_groupFeatureIndex >= 0 && !_groups.empty())
    {
        return scoreGrouped(rows, quantized);
    }
    if(!_earlyExit.empty())
    {
        return scoreStaged(rows.count, quantized);
    }
    BatchScores scores;
    scores.raw.reserve(rows.count);
    for(size_t i = 0; i < rows.count; ++i)
    {
        scores.raw.push_back(scoreRoots(quantized.data() + (i * width), _roots));
    }
    return scores;
}

inline BatchScores TreeDataAdapter::scoreStaged(size_t count,
                                                const std::vector<uint32_t>& quantized) const
{
    const size_t width = quantizedWidth();
    BatchScores scores;
    scores.raw.assign(count, 0.0);
    scores.stage.assign(count, 0);
    scores.stages = static_cast<uint8_t>(_earlyExit.size());

    // Partial sums accumulate tree by tree in model order, so a survivor's final sum is
    // the one a full walk computes.
    std::vector<size_t> alive(count);
    std::iota(alive.begin(), alive.end(), size_t{0});
    const auto better = [&](size_t a, size_t b) {
        const double left = scores.raw[a];
        const double right = scores.raw[b];
        if(left != right)
        {
            return _lowerIsBetter ? left < right : left > right;
        }
        return a < b; // Ties keep the earlier row, so pruning does not depend on the sort.
    };
    size_t done = 0;
    for(size_t s = 0; s < _earlyExit.size(); ++s)
    {
        const auto& stage = _earlyExit[s];
        for(const size_t i : alive)
        {
            accumulate(quantized.data() + (i * width), _roots, done, stage.trees, scores.raw[i]);
        }
        done = stage.trees;
        if(alive.size() > stage.keep)
        {
            std::nth_element(alive.begin(), alive.begin() + stage.keep, alive.end(), better);
            // Rows past `keep` were pruned after surviving s stages; their partial sums stay.
            for(size_t j = stage.keep; j < alive.size(); ++j)
            {
                scores.stage[alive[j]] = static_cast<uint8_t>(s);
            }
            alive.resize(stage.keep);
        }
    }
    for(const size_t i : alive)
    {
        accumulate(quantized.data() + (i * width), _roots, done, _roots.size(), scores.raw[i]);
        // Add the bias last, as scoreRoots does.
        scores.raw[i] = _baseScore + scores.raw[i];
        scores.stage[i] = scores.stages;
    }
    return scores;
}

inline BatchScores TreeDataAdapter::scoreGrouped(const FeatureRows& rows,
                                                 const std::vector<uint32_t>& quantized) const
{
    const size_t width = quantizedWidth();
    BatchScores scores;
    scores.raw.assign(rows.count, -std::numeric_limits<double>::infinity());

    // Layer 1: a group's standing is the best layer-1 score among its rows, best in the
    // objective's direction. Raw scores are compared; every transform is increasing.
    // Only scores RFC 0019 §8.3 admits count. Ties go to the smaller group value so the
    // choice does not depend on row order.
    const auto slot = static_cast<size_t>(_groupFeatureIndex);
    if(slot >= rows.width)
    {
        return scores;
    }
    double bestGroupScore = 0.0;
    double chosenGroup = 0.0;
    bool chosen = false;
    for(size_t i = 0; i < rows.count; ++i)
    {
        const double group = rows.row(i)[slot];
        // Nonfinite group values cannot name an admitted group.
        if(!std::isfinite(group))
        {
            continue;
        }
        const double groupScore = scoreRoots(quantized.data() + (i * width), _roots);
        if(!score_transform::isRankableScore(score_transform::applyInverse(groupScore, _transform),
                                             _positiveRequired))
        {
            continue;
        }
        const bool better
            = _lowerIsBetter ? groupScore < bestGroupScore : groupScore > bestGroupScore;
        if(!chosen || better || (groupScore == bestGroupScore && group < chosenGroup))
        {
            bestGroupScore = groupScore;
            chosenGroup = group;
            chosen = true;
        }
    }
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
    for(size_t i = 0; i < rows.count; ++i)
    {
        if(rows.row(i)[slot] != chosenGroup)
        {
            continue;
        }
        // Chosen group missing from layer 2: fall back to layer 1 rather than reject it.
        scores.raw[i]
            = scoreRoots(quantized.data() + (i * width),
                         within == nullptr || within->roots.empty() ? _roots : within->roots);
    }
    return scores;
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
