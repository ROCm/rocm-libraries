// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "IUhdAdapter.hpp"

#include <memory>
#include <string>
#include <vector>

// Forward declare FlatBuffer types
namespace hipdnn_flatbuffers_sdk::data_objects
{
struct GbdtModel;
struct GbdtTree;
} // namespace hipdnn_flatbuffers_sdk::data_objects

namespace hipdnn_backend::heuristics::uhd
{

/// @brief GBDT tree walker adapter for scoring kernel candidates.
///
/// Loads a GBDT model from FlatBuffer format and evaluates it on feature vectors.
/// Validates that the model's features_hash matches the UHD's signature hash.
class TreeDataAdapter : public IUhdAdapter
{
public:
    /// Load a GBDT model from a FlatBuffer file.
    /// @param modelPath Path to the .fb model file.
    /// @param expectedFeaturesHash Hash from UHD features_signature.
    /// @returns Adapter or nullptr if loading/validation fails.
    static std::unique_ptr<TreeDataAdapter> load(const std::string& modelPath,
                                                 const std::string& expectedFeaturesHash);

    /// Load from an in-memory buffer.
    /// @param buffer Pointer to FlatBuffer data (must remain valid).
    /// @param size Size of buffer in bytes.
    /// @param expectedFeaturesHash Hash from UHD features_signature.
    /// @returns Adapter or nullptr if validation fails.
    static std::unique_ptr<TreeDataAdapter>
        loadFromBuffer(const uint8_t* buffer, size_t size, const std::string& expectedFeaturesHash);

    ~TreeDataAdapter() override;

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
        return _treeCount;
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
    /// Returns true if training_arches is empty (no restriction) or if arch is in the list.
    bool isTrainedForArch(const std::string& arch) const override;

private:
    TreeDataAdapter(std::vector<uint8_t> ownedBuffer,
                    const hipdnn_flatbuffers_sdk::data_objects::GbdtModel* model,
                    std::string featuresHash,
                    size_t numFeatures,
                    double baseScore,
                    double learningRate,
                    std::vector<std::string> trainingArches,
                    std::string modelVersion);

    /// Evaluate a single tree.
    static double evaluateTree(const hipdnn_flatbuffers_sdk::data_objects::GbdtTree* tree,
                               const std::vector<double>& features);

    std::vector<uint8_t> _ownedBuffer;
    const hipdnn_flatbuffers_sdk::data_objects::GbdtModel* _model;
    std::string _featuresHash;
    size_t _numFeatures;
    size_t _treeCount;
    double _baseScore;
    // NOTE: _learningRate is stored for metadata/debugging but NOT used in score().
    // LightGBM's dump_model() returns leaf_values that already include the learning_rate.
    [[maybe_unused]] double _learningRate;

    // RFC 0019 §9.2, §13: Model provenance for out-of-distribution detection
    std::vector<std::string> _trainingArches;
    std::string _modelVersion;
};

} // namespace hipdnn_backend::heuristics::uhd
