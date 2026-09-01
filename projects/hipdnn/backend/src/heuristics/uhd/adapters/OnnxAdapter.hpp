// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_plugin_sdk/ingestor/uhd/adapters/IUhdAdapter.hpp>

#include <memory>
#include <string>
#include <vector>

#ifdef HIPDNN_ENABLE_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace hipdnn_backend::heuristics::uhd
{

// Names now come straight from the plugin SDK; the local forwarding headers that
// used to alias them are gone (RFC 0019 §5 puts this machinery in the engine).
using hipdnn_plugin_sdk::ingestor::uhd::IUhdAdapter;
using hipdnn_plugin_sdk::ingestor::uhd::UhdAdapterType;


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
