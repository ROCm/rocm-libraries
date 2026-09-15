// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file GbdtModelTestBuilder.hpp
 * @brief In-memory GBDT FlatBuffer builder shared by the UHD tests.
 *
 * Lets a test construct a tree_data model artifact without a training run, so both
 * the adapter tests and the end-to-end selection-flow tests can exercise real models
 * (including training-arch metadata) from the same builder.
 */

#include <hipdnn_flatbuffers_sdk/data_objects/gbdt_model_generated.h>

#include <cstdint>
#include <fstream>
#include <string>
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

    GbdtModelTestBuilder& addTree(const TreeSpec& tree)
    {
        _trees.push_back(tree);
        return *this;
    }

    /// Build and return the FlatBuffer data.
    std::vector<uint8_t> build()
    {
        flatbuffers::FlatBufferBuilder fbb;

        // Build trees
        std::vector<flatbuffers::Offset<fb::GbdtTree>> treeOffsets;
        for(const auto& tree : _trees)
        {
            auto treeOffset = fb::CreateGbdtTreeDirect(fbb,
                                                       &tree.featureIndices,
                                                       &tree.thresholds,
                                                       &tree.leftChildren,
                                                       &tree.rightChildren,
                                                       &tree.leafValues,
                                                       &tree.defaultLeft);
            treeOffsets.push_back(treeOffset);
        }

        auto treesVector = fbb.CreateVector(treeOffsets);
        auto hashOffset = fbb.CreateString(_featuresHash);

        // Build training_arches vector if provided
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

        // Build model_version if provided
        flatbuffers::Offset<flatbuffers::String> versionOffset = 0;
        if(!_modelVersion.empty())
        {
            versionOffset = fbb.CreateString(_modelVersion);
        }

        fb::GbdtModelBuilder modelBuilder(fbb);
        modelBuilder.add_trees(treesVector);
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

private:
    int32_t _numFeatures = 0;
    std::string _featuresHash;
    double _baseScore = 0.0;
    double _learningRate = 1.0;
    std::vector<TreeSpec> _trees;
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
