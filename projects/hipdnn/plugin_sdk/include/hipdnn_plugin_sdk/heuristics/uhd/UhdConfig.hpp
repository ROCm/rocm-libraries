// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// @file UhdConfig.hpp
/// @brief The resolved contents of one UHD, as the loader produces it.
namespace hipdnn_plugin_sdk::uhd
{

/// The resolved feature contract and scorer configuration owned by one engine UHD.
struct UhdConfig
{
    std::string uhdId;
    std::string name;

    /// Backfilled by the descriptor loader from the UED role map (RFC 0019 §3.1); empty
    /// outside a descriptor set.
    std::string engineName;
    std::string role;
    std::string arch;
    /// The descriptor set this UHD was generated against: `ued`, `kmd` and `umd`.
    nlohmann::json trainedAgainst;

    std::vector<nlohmann::json> featuresSignature;
    std::string featuresHash;
    std::string objective = "max"; // "max" or "min"

    // Registered ranking metric (RankingMetrics.hpp; RFC 0019 §4.4); empty for a ranker
    // whose scores are not comparable predictions.
    std::string scoreMetric; // e.g., "tflops", "time"
    bool scoreCalibrated = false; // cross-engine comparable?
    std::string scoreTransform; // e.g., "log1p", "identity"

    // Adapter configuration
    std::string adapterType = "static_order"; // "static_order", "tree_data", etc.
    std::string modelArtifactPath; // for tree_data/onnx/custom_library
    std::string modelHash; // artifact SHA-256: declared, else of the bytes present at parse
    std::string customLibrarySymbol; // for custom_library: symbol name in .so
    std::string nativeSymbol; // for native: symbol registered with NativeScorerRegistry

    /// RFC 0019 §6.5: field -> (string value -> code). Empty must hash identically to a UHD
    /// without the field.
    std::map<std::string, std::map<std::string, int32_t>> categoricalEncoding;
};

} // namespace hipdnn_plugin_sdk::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
