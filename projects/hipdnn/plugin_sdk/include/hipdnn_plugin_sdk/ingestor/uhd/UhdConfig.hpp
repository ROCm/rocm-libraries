// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <map>
#include <string>
#include <utility>
#include <vector>

/// @file UhdConfig.hpp
/// @brief The resolved contents of one UHD, as the loader produces it.
///
/// Kept apart from any registry so both the policy path and the ingestor's
/// plan-build path can name it without depending on how the other finds engines.
namespace hipdnn_plugin_sdk::ingestor::uhd
{

/// @brief UHD configuration for an engine (mock for UED-owned UHD).
///
/// Contains the heuristic configuration: features signature, adapter,
/// objective, and score metadata. In full RFC 0017 integration, this
/// is owned by the UED.
struct UhdConfig
{
    std::string uhdId;
    std::string name;

    // RFC 0019 §6.4: Derived values (ordered map of name → JsonLogic expression)
    std::vector<std::pair<std::string, std::string>> derived; // name, expression (JSON string)

    std::vector<std::string> featuresSignature;
    std::string featuresHash;
    std::string objective = "max"; // "max" or "min"

    // Score metadata for cross-engine comparison (RFC §5, §12.3)
    std::string scoreUnits; // e.g., "tflops", "ms"
    bool scoreCalibrated = false; // cross-engine comparable?
    std::string scoreTransform; // e.g., "log1p", "identity"

    // Adapter configuration
    std::string adapterType = "static_order"; // "static_order", "tree_data", etc.
    std::string modelArtifactPath; // for tree_data/onnx/custom_library
    std::string modelHash; // checksum of model artifact for integrity validation
    std::vector<std::string> staticOrderFields = {"priority", "id"}; // for static_order
    std::string customLibrarySymbol; // for custom_library: symbol name in .so
    std::string nativeSymbol; // for native: symbol registered with NativeScorerRegistry

    /// RFC 0019 §6.5: field -> (string value -> code), for features that read a string field.
    /// Empty when none does, which is the common case and must hash identically to a UHD
    /// that predates the field.
    std::map<std::string, std::map<std::string, int32_t>> categoricalEncoding;
};

} // namespace hipdnn_plugin_sdk::ingestor::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
