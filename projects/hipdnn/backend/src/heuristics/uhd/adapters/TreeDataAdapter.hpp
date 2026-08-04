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
    static std::unique_ptr<TreeDataAdapter> loadFromBuffer(const uint8_t* buffer,
                                                            size_t size,
                                                            const std::string& expectedFeaturesHash);

    ~TreeDataAdapter() override;

    double score(const std::vector<double>& features) const override;

    UhdAdapterType type() const override { return UhdAdapterType::TreeData; }

    size_t expectedFeatureCount() const override { return _numFeatures; }

    const std::string& getFeaturesHash() const override { return _featuresHash; }

    /// Get the number of trees in the ensemble.
    size_t treeCount() const { return _treeCount; }

private:
    TreeDataAdapter(std::vector<uint8_t> ownedBuffer,
                    const hipdnn_flatbuffers_sdk::data_objects::GbdtModel* model,
                    std::string featuresHash,
                    size_t numFeatures,
                    double baseScore,
                    double learningRate);

    /// Evaluate a single tree.
    double evaluateTree(const hipdnn_flatbuffers_sdk::data_objects::GbdtTree* tree,
                        const std::vector<double>& features) const;

    std::vector<uint8_t> _ownedBuffer;
    const hipdnn_flatbuffers_sdk::data_objects::GbdtModel* _model;
    std::string _featuresHash;
    size_t _numFeatures;
    size_t _treeCount;
    double _baseScore;
    double _learningRate;
};

} // namespace hipdnn_backend::heuristics::uhd
