// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// @file UhdConfig.hpp
/// @brief The resolved contents of one UHD, as the loader produces it.
///
/// Kept apart from any registry so both the policy path and the ingestor's
/// plan-build path can name it without depending on how the other finds engines.
namespace hipdnn_plugin_sdk::uhd
{

/// The resolved feature contract and scorer configuration owned by one engine UHD.
struct UhdConfig
{
    std::string uhdId;
    std::string name;

    /// Backfilled by the descriptor loader from the UED role map that resolved this
    /// UHD (RFC 0019 §3.1); never authored in the document itself. Empty for a config
    /// parsed outside a descriptor set.
    std::string engineName;
    std::string role;
    std::string arch;
    /// The descriptor set this UHD was generated against: `ued`, `kmd` and `umd`.
    nlohmann::json trainedAgainst;

    std::vector<nlohmann::json> featuresSignature;
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

} // namespace hipdnn_plugin_sdk::uhd
