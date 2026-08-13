// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "IUhdAdapter.hpp"

#include <memory>
#include <string>
#include <vector>

namespace hipdnn_backend::heuristics::uhd
{

/// @brief ONNX Runtime adapter for neural network models (RFC 0019 §7.3).
///
/// The ONNX adapter is an opt-in, dependency-gated adapter that loads ONNX
/// models via ONNX Runtime. It requires the onnxruntime library to be
/// available at load time.
///
/// Current implementation status: DEPENDENCY-GATED STUB
/// - ONNX Runtime is not available in the current build environment
/// - load() returns nullptr with a clear log message
/// - This satisfies the contract gap (ONNX is schema-valid, not silently rejected)
/// - Will be implemented fully when ONNX Runtime dependency is added
///
/// Design (when implemented):
/// - Load .onnx model file via Ort::Session
/// - Input: feature vector as 1D tensor
/// - Output: single float score
/// - Session cached per process (similar to other adapters)
class OnnxAdapter : public IUhdAdapter
{
public:
    /// @brief Load an ONNX model (stub - returns nullptr until dependency available).
    ///
    /// @param modelPath Absolute path to the .onnx file.
    /// @param expectedFeaturesHash SHA-256 hash of the feature signature.
    /// @return nullptr (ONNX Runtime not available in current build)
    ///
    /// TODO: Implement when ONNX Runtime dependency is added:
    /// - Link against onnxruntime library
    /// - Create Ort::Env and Ort::Session
    /// - Verify input shape matches num_features
    /// - Cache session for reuse
    static std::unique_ptr<OnnxAdapter> load(const std::string& modelPath,
                                              const std::string& expectedFeaturesHash);

    double score(const std::vector<double>& features) const override;
    UhdAdapterType type() const override
    {
        return UhdAdapterType::ONNX;
    }
    size_t expectedFeatureCount() const override
    {
        return _numFeatures;
    }
    const std::string& getFeaturesHash() const override
    {
        return _featuresHash;
    }

private:
    OnnxAdapter(size_t numFeatures, std::string featuresHash);

    size_t _numFeatures;
    std::string _featuresHash;
    // TODO: Add Ort::Session* _session when ONNX Runtime is available
};

} // namespace hipdnn_backend::heuristics::uhd
