// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "IUhdAdapter.hpp"

#include <memory>
#include <string>
#include <vector>

#ifdef HIPDNN_ENABLE_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace hipdnn_backend::heuristics::uhd
{

/// @brief ONNX Runtime adapter for neural network models (RFC 0019 §7.3).
///
/// The ONNX adapter is an opt-in, dependency-gated adapter that loads ONNX
/// models via ONNX Runtime. It requires the onnxruntime library to be
/// available at compile time (HIPDNN_ENABLE_ONNX defined) and link time.
///
/// When HIPDNN_ENABLE_ONNX is not defined:
/// - load() returns nullptr with a clear log message
/// - This satisfies the contract gap (ONNX is schema-valid, not silently rejected)
///
/// When HIPDNN_ENABLE_ONNX is defined:
/// - Load .onnx model file via Ort::Session
/// - Input: feature vector as 1D tensor (batch=1, features=N)
/// - Output: single float score
/// - Session cached per adapter instance
class OnnxAdapter : public IUhdAdapter
{
public:
    /// @brief Load an ONNX model from disk.
    ///
    /// Returns nullptr (with error log) if:
    /// - ONNX Runtime is not available (HIPDNN_ENABLE_ONNX not defined)
    /// - Model file does not exist
    /// - Model is not a valid .onnx file
    /// - Input shape is incompatible (not 2D with batch=1)
    ///
    /// @param modelPath Absolute path to the .onnx file.
    /// @param expectedFeaturesHash SHA-256 hash of the feature signature (informational).
    /// @return Adapter on success, nullptr on failure.
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

#ifdef HIPDNN_ENABLE_ONNX
    static Ort::Env& getEnv();
#endif

    size_t _numFeatures;
    std::string _featuresHash;

#ifdef HIPDNN_ENABLE_ONNX
    std::shared_ptr<Ort::Session> _session;
    std::string _inputName;
    std::string _outputName;
#endif
};

} // namespace hipdnn_backend::heuristics::uhd
