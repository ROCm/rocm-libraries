// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "OnnxAdapter.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipdnn_backend::heuristics::uhd
{

std::unique_ptr<OnnxAdapter> OnnxAdapter::load(const std::string& modelPath,
                                                const std::string& expectedFeaturesHash)
{
    // ONNX adapter is dependency-gated per RFC 0019 §7.3.
    // ONNX Runtime is not available in the current build environment.
    // This stub satisfies the contract (ONNX is schema-valid, not silently rejected)
    // and will be implemented fully when the dependency is added.
    //
    // When implementing:
    // 1. Check for onnxruntime library availability (dlopen or link-time)
    // 2. Create Ort::Env (process-global, cached)
    // 3. Create Ort::SessionOptions with appropriate providers (CPU, ROCM)
    // 4. Load Ort::Session from modelPath
    // 5. Verify input tensor shape matches expected feature count
    // 6. Verify features_hash if model embeds it
    // 7. Return adapter with cached session

    (void)modelPath;              // Suppress unused parameter warning
    (void)expectedFeaturesHash;   // Suppress unused parameter warning

    HIPDNN_SDK_LOG_ERROR("OnnxAdapter: ONNX Runtime dependency not available. "
                         << "The ONNX adapter is opt-in and dependency-gated per RFC 0019 §7.3. "
                         << "To use ONNX models, build with ONNX Runtime support. "
                         << "Model path: " << modelPath);
    return nullptr;
}

OnnxAdapter::OnnxAdapter(size_t numFeatures, std::string featuresHash)
    : _numFeatures(numFeatures), _featuresHash(std::move(featuresHash))
{
}

double OnnxAdapter::score(const std::vector<double>& features) const
{
    // This should never be called since load() returns nullptr.
    // If it somehow is called, throw rather than returning a bogus score.
    (void)features;
    throw std::logic_error(
        "OnnxAdapter::score() called on stub adapter. ONNX Runtime not available.");
}

} // namespace hipdnn_backend::heuristics::uhd
