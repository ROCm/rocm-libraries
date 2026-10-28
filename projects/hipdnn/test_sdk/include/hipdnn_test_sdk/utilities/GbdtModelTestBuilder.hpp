// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file GbdtModelTestBuilder.hpp
 * @brief In-memory GBDT FlatBuffer builder shared by the UHD tests.
 */

#include <hipdnn_flatbuffers_sdk/data_objects/gbdt_model_generated.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

/// Helper to build a GBDT model FlatBuffer in memory.
class GbdtModelTestBuilder
{
public:
    struct TreeSpec
    {
        std::vector<int32_t> featureIndices;
        std::vector<double> thresholds;
        std::vector<int32_t> leftChildren;
        std::vector<int32_t> rightChildren;
        std::vector<double> leafValues;
        std::vector<uint8_t> defaultLeft;
    };

    /// A tree in the split/leaf encoding, written exactly as given (gbdt_model.fbs): children
    /// c >= 0 are splits, c < 0 leaf ~c; threshold ids index the model's threshold table.
    struct SplitLeafSpec
    {
        std::vector<int32_t> features;
        std::vector<int32_t> thresholdIds;
        std::vector<int32_t> lefts;
        std::vector<int32_t> rights;
        std::vector<double> leaves;
        std::optional<std::vector<uint8_t>> defaultLeft;
        std::optional<std::vector<uint8_t>> decisionLte;
        bool defaultLeftFill = false;
        /// Storage per array; NONE picks the narrowest type that holds the values.
        fb::IntegerArray featuresType = fb::IntegerArray::NONE;
        fb::IntegerArray thresholdIdsType = fb::IntegerArray::NONE;
        fb::IntegerArray childrenType = fb::IntegerArray::NONE;
        fb::RealArray leavesType = fb::RealArray::Float64Array;
    };

    GbdtModelTestBuilder& setNumFeatures(int32_t n)
    {
        _numFeatures = n;
        return *this;
    }

    GbdtModelTestBuilder& setFeaturesHash(const std::string& hash)
    {
        _featuresHash = hash;
        return *this;
    }

    GbdtModelTestBuilder& setBaseScore(double score)
    {
        _baseScore = score;
        return *this;
    }

    GbdtModelTestBuilder& setLearningRate(double rate)
    {
        _learningRate = rate;
        return *this;
    }

    GbdtModelTestBuilder& setTrainingArches(const std::vector<std::string>& arches)
    {
        _trainingArches = arches;
        return *this;
    }

    GbdtModelTestBuilder& setModelVersion(const std::string& version)
    {
        _modelVersion = version;
        return *this;
    }

    /// Feature slot carrying the group for layer 2 (RFC 0019 grouped tree_data). Without it the
    /// model stays single-layer.
    GbdtModelTestBuilder& setGroupByFeatureIndex(int32_t index)
    {
        _groupByFeatureIndex = index;
        return *this;
    }

    GbdtModelTestBuilder& addGroup(double value, const std::vector<TreeSpec>& trees)
    {
        _groups.emplace_back(value, trees);
        return *this;
    }

    GbdtModelTestBuilder& addTree(const TreeSpec& tree)
    {
        _trees.emplace_back(tree);
        return *this;
    }

    /// A tree written as given in the split/leaf encoding, with setThresholdTable's table.
    GbdtModelTestBuilder& addSplitLeafTree(const SplitLeafSpec& tree)
    {
        _trees.emplace_back(tree);
        return *this;
    }

    /// Write every node-array tree (layer 1 and groups) in the split/leaf encoding instead,
    /// thresholds in one table, leaves as @p leavesType: the same model, encoded the other way.
    GbdtModelTestBuilder& encodeAsSplitLeaf(fb::RealArray leavesType = fb::RealArray::Float64Array,
                                            fb::IntegerArray integerType = fb::IntegerArray::NONE)
    {
        _splitLeaf = true;
        _leavesType = leavesType;
        _integerType = integerType;
        return *this;
    }

    /// The table split/leaf threshold ids index, stored as @p type. Unset, encodeAsSplitLeaf
    /// builds it from the node-array trees' thresholds.
    GbdtModelTestBuilder& setThresholdTable(const std::vector<double>& table,
                                            fb::RealArray type = fb::RealArray::Float64Array)
    {
        _thresholdTable = table;
        _thresholdTableType = type;
        return *this;
    }

    /// Store the table's tag as @p type without changing its contents: a tag a reader may not
    /// know, for the refusal tests.
    GbdtModelTestBuilder& overrideThresholdTableTag(fb::RealArray type)
    {
        _thresholdTableTagOverride = type;
        return *this;
    }

    /// @p tree, a node-array tree whose node 0 is its root, in the split/leaf encoding:
    /// splits numbered in node order (so the root is split 0), leaves likewise.
    static SplitLeafSpec toSplitLeaf(const TreeSpec& tree, const std::vector<double>& table)
    {
        SplitLeafSpec out;
        std::vector<int32_t> renumbered(tree.leftChildren.size());
        int32_t splits = 0;
        int32_t leaves = 0;
        for(size_t i = 0; i < tree.leftChildren.size(); ++i)
        {
            renumbered[i] = tree.leftChildren[i] == -1 ? ~leaves++ : splits++;
        }
        for(size_t i = 0; i < tree.leftChildren.size(); ++i)
        {
            if(tree.leftChildren[i] == -1)
            {
                out.leaves.push_back(i < tree.leafValues.size() ? tree.leafValues[i] : 0.0);
                continue;
            }
            out.features.push_back(tree.featureIndices[i]);
            const auto at = std::find(table.begin(), table.end(), tree.thresholds[i]);
            out.thresholdIds.push_back(static_cast<int32_t>(at - table.begin()));
            out.lefts.push_back(renumbered[static_cast<size_t>(tree.leftChildren[i])]);
            out.rights.push_back(renumbered[static_cast<size_t>(tree.rightChildren[i])]);
        }
        std::vector<uint8_t> defaultLeft;
        for(size_t i = 0; i < tree.leftChildren.size(); ++i)
        {
            if(tree.leftChildren[i] != -1)
            {
                defaultLeft.push_back(i < tree.defaultLeft.size() ? tree.defaultLeft[i] : 0);
            }
        }
        out.defaultLeft = defaultLeft;
        return out;
    }

    /// Build and return the FlatBuffer data.
    std::vector<uint8_t> build()
    {
        flatbuffers::FlatBufferBuilder fbb;

        std::vector<double> table = _thresholdTable;
        if(_splitLeaf && table.empty())
        {
            for(const auto& tree : allNodeArrayTrees())
            {
                for(size_t i = 0; i < tree.leftChildren.size(); ++i)
                {
                    if(tree.leftChildren[i] != -1)
                    {
                        table.push_back(tree.thresholds[i]);
                    }
                }
            }
            std::sort(table.begin(), table.end());
            table.erase(std::unique(table.begin(), table.end()), table.end());
        }

        std::vector<flatbuffers::Offset<fb::GbdtTree>> treeOffsets;
        for(const auto& tree : _trees)
        {
            treeOffsets.push_back(std::holds_alternative<TreeSpec>(tree)
                                      ? writeTree(fbb, std::get<TreeSpec>(tree), table)
                                      : writeSplitLeaf(fbb, std::get<SplitLeafSpec>(tree)));
        }

        auto treesVector = fbb.CreateVector(treeOffsets);
        auto hashOffset = fbb.CreateString(_featuresHash);

        flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>>
            archesVectorOffset = 0;
        if(!_trainingArches.empty())
        {
            std::vector<flatbuffers::Offset<flatbuffers::String>> archOffsets;
            archOffsets.reserve(_trainingArches.size());
            for(const auto& arch : _trainingArches)
            {
                archOffsets.push_back(fbb.CreateString(arch));
            }
            archesVectorOffset = fbb.CreateVector(archOffsets);
        }

        flatbuffers::Offset<flatbuffers::String> versionOffset = 0;
        if(!_modelVersion.empty())
        {
            versionOffset = fbb.CreateString(_modelVersion);
        }

        // Groups are serialised before GbdtModelBuilder opens: nested tables cannot be
        // created while a table is under construction.
        std::vector<flatbuffers::Offset<fb::GbdtGroup>> groupOffsets;
        for(const auto& [value, trees] : _groups)
        {
            std::vector<flatbuffers::Offset<fb::GbdtTree>> inner;
            for(const auto& tree : trees)
            {
                inner.push_back(writeTree(fbb, tree, table));
            }
            groupOffsets.push_back(fb::CreateGbdtGroup(fbb, value, fbb.CreateVector(inner)));
        }
        const auto groupsVector = groupOffsets.empty() ? 0 : fbb.CreateVector(groupOffsets).o;
        const bool hasTable = !table.empty() || _thresholdTableTagOverride.has_value();
        const auto tableOffset = hasTable ? writeReals(fbb, table, _thresholdTableType).o : 0;

        fb::GbdtModelBuilder modelBuilder(fbb);
        modelBuilder.add_trees(treesVector);
        if(_groupByFeatureIndex >= 0)
        {
            modelBuilder.add_group_by_feature_index(_groupByFeatureIndex);
        }
        if(groupsVector != 0)
        {
            modelBuilder.add_groups(
                flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<fb::GbdtGroup>>>(
                    groupsVector));
        }
        modelBuilder.add_num_features(_numFeatures);
        modelBuilder.add_features_hash(hashOffset);
        modelBuilder.add_base_score(_baseScore);
        modelBuilder.add_learning_rate(_learningRate);
        if(!_trainingArches.empty())
        {
            modelBuilder.add_training_arches(archesVectorOffset);
        }
        if(!_modelVersion.empty())
        {
            modelBuilder.add_model_version(versionOffset);
        }
        if(hasTable)
        {
            modelBuilder.add_threshold_table_type(
                _thresholdTableTagOverride.value_or(_thresholdTableType));
            modelBuilder.add_threshold_table(flatbuffers::Offset<void>(tableOffset));
        }

        auto modelOffset = modelBuilder.Finish();
        fbb.Finish(modelOffset, fb::GbdtModelIdentifier());

        return {fbb.GetBufferPointer(), fbb.GetBufferPointer() + fbb.GetSize()};
    }

    /// Build and write to disk, for tests that need a model artifact path.
    /// @returns true if the file was written.
    bool buildToFile(const std::string& path)
    {
        const auto buffer = build();
        std::ofstream out(path, std::ios::binary);
        if(!out)
        {
            return false;
        }
        out.write(reinterpret_cast<const char*>(buffer.data()),
                  static_cast<std::streamsize>(buffer.size()));
        return out.good();
    }

    /// The narrowest IntegerArray type holding every value, as uhd_gen's converter picks.
    static fb::IntegerArray narrowest(const std::vector<int32_t>& values)
    {
        const auto [low, high]
            = values.empty()
                  ? std::pair<int32_t, int32_t>{0, 0}
                  : std::pair<int32_t, int32_t>{*std::min_element(values.begin(), values.end()),
                                                *std::max_element(values.begin(), values.end())};
        if(low >= 0 && high <= std::numeric_limits<uint8_t>::max())
        {
            return fb::IntegerArray::UInt8Array;
        }
        if(low >= std::numeric_limits<int8_t>::min() && high <= std::numeric_limits<int8_t>::max())
        {
            return fb::IntegerArray::Int8Array;
        }
        if(low >= 0 && high <= std::numeric_limits<uint16_t>::max())
        {
            return fb::IntegerArray::UInt16Array;
        }
        if(low >= std::numeric_limits<int16_t>::min()
           && high <= std::numeric_limits<int16_t>::max())
        {
            return fb::IntegerArray::Int16Array;
        }
        return fb::IntegerArray::Int32Array;
    }

private:
    template <typename T>
    static std::vector<T> narrowTo(const std::vector<int32_t>& values)
    {
        return std::vector<T>(values.begin(), values.end());
    }

    /// @p values stored as @p type (NONE: the narrowest that holds them). A tag outside the
    /// union is written with Int32 contents, for the unknown-tag tests.
    static std::pair<fb::IntegerArray, flatbuffers::Offset<void>>
        writeIntegers(flatbuffers::FlatBufferBuilder& fbb,
                      const std::vector<int32_t>& values,
                      fb::IntegerArray type)
    {
        if(type == fb::IntegerArray::NONE)
        {
            type = narrowest(values);
        }
        switch(type)
        {
        case fb::IntegerArray::Int8Array:
        {
            const auto narrowed = narrowTo<int8_t>(values);
            return {type, fb::CreateInt8ArrayDirect(fbb, &narrowed).Union()};
        }
        case fb::IntegerArray::UInt8Array:
        {
            const auto narrowed = narrowTo<uint8_t>(values);
            return {type, fb::CreateUInt8ArrayDirect(fbb, &narrowed).Union()};
        }
        case fb::IntegerArray::Int16Array:
        {
            const auto narrowed = narrowTo<int16_t>(values);
            return {type, fb::CreateInt16ArrayDirect(fbb, &narrowed).Union()};
        }
        case fb::IntegerArray::UInt16Array:
        {
            const auto narrowed = narrowTo<uint16_t>(values);
            return {type, fb::CreateUInt16ArrayDirect(fbb, &narrowed).Union()};
        }
        default:
            return {type, fb::CreateInt32ArrayDirect(fbb, &values).Union()};
        }
    }

    static flatbuffers::Offset<void> writeReals(flatbuffers::FlatBufferBuilder& fbb,
                                                const std::vector<double>& values,
                                                fb::RealArray type)
    {
        if(type == fb::RealArray::Float32Array)
        {
            const std::vector<float> narrowed(values.begin(), values.end());
            return fb::CreateFloat32ArrayDirect(fbb, &narrowed).Union();
        }
        return fb::CreateFloat64ArrayDirect(fbb, &values).Union();
    }

    flatbuffers::Offset<fb::GbdtTree> writeTree(flatbuffers::FlatBufferBuilder& fbb,
                                                const TreeSpec& tree,
                                                const std::vector<double>& table) const
    {
        if(!_splitLeaf)
        {
            return fb::CreateGbdtTreeDirect(fbb,
                                            &tree.featureIndices,
                                            &tree.thresholds,
                                            &tree.leftChildren,
                                            &tree.rightChildren,
                                            &tree.leafValues,
                                            &tree.defaultLeft);
        }
        auto spec = toSplitLeaf(tree, table);
        spec.leavesType = _leavesType;
        spec.featuresType = spec.thresholdIdsType = spec.childrenType = _integerType;
        return writeSplitLeaf(fbb, spec);
    }

    static flatbuffers::Offset<fb::GbdtTree> writeSplitLeaf(flatbuffers::FlatBufferBuilder& fbb,
                                                            const SplitLeafSpec& tree)
    {
        // Nested tables are written before the tree's own builder opens.
        std::vector<int32_t> children = tree.lefts;
        children.insert(children.end(), tree.rights.begin(), tree.rights.end());
        const auto childType
            = tree.childrenType == fb::IntegerArray::NONE ? narrowest(children) : tree.childrenType;
        const bool hasSplits = !tree.lefts.empty() || !tree.features.empty();
        std::pair<fb::IntegerArray, flatbuffers::Offset<void>> features{};
        std::pair<fb::IntegerArray, flatbuffers::Offset<void>> ids{};
        std::pair<fb::IntegerArray, flatbuffers::Offset<void>> lefts{};
        std::pair<fb::IntegerArray, flatbuffers::Offset<void>> rights{};
        if(hasSplits)
        {
            features = writeIntegers(fbb, tree.features, tree.featuresType);
            ids = writeIntegers(fbb, tree.thresholdIds, tree.thresholdIdsType);
            lefts = writeIntegers(fbb, tree.lefts, childType);
            rights = writeIntegers(fbb, tree.rights, childType);
        }
        const auto defaultLeft = tree.defaultLeft ? fbb.CreateVector(*tree.defaultLeft).o : 0;
        const auto decisionLte = tree.decisionLte ? fbb.CreateVector(*tree.decisionLte).o : 0;
        const auto leaves = writeReals(fbb, tree.leaves, tree.leavesType);

        fb::GbdtTreeBuilder builder(fbb);
        if(hasSplits)
        {
            builder.add_split_features_type(features.first);
            builder.add_split_features(features.second);
            builder.add_split_thresholds_type(ids.first);
            builder.add_split_thresholds(ids.second);
            builder.add_split_left_type(lefts.first);
            builder.add_split_left(lefts.second);
            builder.add_split_right_type(rights.first);
            builder.add_split_right(rights.second);
        }
        if(defaultLeft != 0)
        {
            builder.add_split_default_left(
                flatbuffers::Offset<flatbuffers::Vector<uint8_t>>(defaultLeft));
        }
        if(decisionLte != 0)
        {
            builder.add_split_decision_lte(
                flatbuffers::Offset<flatbuffers::Vector<uint8_t>>(decisionLte));
        }
        builder.add_default_left_fill(tree.defaultLeftFill);
        builder.add_leaves_type(tree.leavesType);
        builder.add_leaves(leaves);
        return builder.Finish();
    }

    std::vector<TreeSpec> allNodeArrayTrees() const
    {
        std::vector<TreeSpec> found;
        for(const auto& tree : _trees)
        {
            if(std::holds_alternative<TreeSpec>(tree))
            {
                found.push_back(std::get<TreeSpec>(tree));
            }
        }
        for(const auto& [value, trees] : _groups)
        {
            found.insert(found.end(), trees.begin(), trees.end());
        }
        return found;
    }

    int32_t _numFeatures = 0;
    std::string _featuresHash;
    double _baseScore = 0.0;
    double _learningRate = 1.0;
    std::vector<std::variant<TreeSpec, SplitLeafSpec>> _trees;
    bool _splitLeaf = false;
    fb::RealArray _leavesType = fb::RealArray::Float64Array;
    fb::IntegerArray _integerType = fb::IntegerArray::NONE;
    std::vector<double> _thresholdTable;
    fb::RealArray _thresholdTableType = fb::RealArray::Float64Array;
    std::optional<fb::RealArray> _thresholdTableTagOverride;
    int32_t _groupByFeatureIndex = -1;
    std::vector<std::pair<double, std::vector<TreeSpec>>> _groups;
    std::vector<std::string> _trainingArches;
    std::string _modelVersion;
};

/// Create a simple single-node tree (just a leaf).
inline GbdtModelTestBuilder::TreeSpec makeLeafTreeSpec(double leafValue)
{
    GbdtModelTestBuilder::TreeSpec spec;
    spec.featureIndices = {0};
    spec.thresholds = {0.0};
    spec.leftChildren = {-1};
    spec.rightChildren = {-1};
    spec.leafValues = {leafValue};
    spec.defaultLeft = {1};
    return spec;
}

} // namespace hipdnn_test_sdk::utilities
