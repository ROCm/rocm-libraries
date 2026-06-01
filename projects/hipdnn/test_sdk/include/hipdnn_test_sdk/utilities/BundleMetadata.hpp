// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#include <hipdnn_data_sdk/logging/Logger.hpp>

#include <nlohmann/json.hpp>

namespace hipdnn_test_sdk::utilities
{

/// Metadata read from a {Name}.meta.json companion file alongside a golden
/// reference bundle ({Name}.json + {Name}.tensor{uid}.bin).
///
/// All fields inside the `metadata` object are optional. A missing field means
/// "not recorded" — the system must behave correctly without it. The only
/// required field is `format_version` at the top level.
struct BundleMetadata
{
    int formatVersion = 1;

    std::optional<std::string> generatorVersion;
    std::optional<std::string> createdAt;
    std::optional<std::string> gpuArchitecture;
    std::optional<std::string> rocmVersion;
    std::optional<std::string> referenceExecutor;
    std::optional<std::string> referenceExecutorHash;
    std::optional<std::string> operation;
    std::optional<int64_t> seed;
    std::optional<int64_t> minimumVramMb;
};

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Derive the .meta.json path from a bundle JSON path.
///   "dir/Small.json" → "dir/Small.meta.json"
inline std::filesystem::path metaJsonPath(const std::filesystem::path& bundleJsonPath)
{
    return bundleJsonPath.parent_path() / (bundleJsonPath.stem().string() + ".meta.json");
}

/// Load metadata from a .meta.json companion file.
///
/// Returns std::nullopt if the file does not exist (backwards compatible — old
/// bundles simply have no metadata). Returns std::nullopt with a warning log
/// if the file exists but cannot be parsed or has an unsupported
/// format_version. Never throws.
inline std::optional<BundleMetadata> loadBundleMetadata(
    const std::filesystem::path& bundleJsonPath)
{
    auto path = metaJsonPath(bundleJsonPath);

    if(!std::filesystem::exists(path))
    {
        HIPDNN_SDK_LOG_INFO("No metadata file found at " << path);
        return std::nullopt;
    }

    try
    {
        std::ifstream file(path);
        if(!file)
        {
            HIPDNN_SDK_LOG_WARN("Could not open metadata file " << path);
            return std::nullopt;
        }

        auto json = nlohmann::json::parse(file);

        if(!json.contains("format_version") || !json["format_version"].is_number_integer())
        {
            HIPDNN_SDK_LOG_WARN(path << " missing or invalid format_version");
            return std::nullopt;
        }

        int version = json["format_version"].get<int>();
        if(version != 1)
        {
            HIPDNN_SDK_LOG_WARN(path << " has unsupported format_version " << version);
            return std::nullopt;
        }

        BundleMetadata meta;
        meta.formatVersion = version;

        if(json.contains("metadata") && json["metadata"].is_object())
        {
            const auto& m = json["metadata"];

            auto readString = [&](const char* key) -> std::optional<std::string> {
                if(m.contains(key) && m[key].is_string())
                {
                    return m[key].get<std::string>();
                }
                return std::nullopt;
            };

            auto readInt64 = [&](const char* key) -> std::optional<int64_t> {
                if(m.contains(key) && m[key].is_number_integer())
                {
                    return m[key].get<int64_t>();
                }
                return std::nullopt;
            };

            meta.generatorVersion = readString("generator_version");
            meta.createdAt = readString("created_at");
            meta.gpuArchitecture = readString("gpu_architecture");
            meta.rocmVersion = readString("rocm_version");
            meta.referenceExecutor = readString("reference_executor");
            meta.referenceExecutorHash = readString("reference_executor_hash");
            meta.operation = readString("operation");
            meta.seed = readInt64("seed");
            meta.minimumVramMb = readInt64("minimum_vram_mb");
        }

        return meta;
    }
    catch(const std::exception& e)
    {
        HIPDNN_SDK_LOG_WARN("Failed to parse metadata file " << path << ": " << e.what());
        return std::nullopt;
    }
}

// ---------------------------------------------------------------------------
// Guard check functions (pure — no HIP, no system calls)
//
// Each function takes metadata + a device-provided value and returns:
//   std::nullopt        → check passed, continue
//   "reason string"     → test should be skipped with this message
//
// To add a new check: write a function with this signature and call it from
// the harness alongside the existing checks.
// ---------------------------------------------------------------------------

/// Skip if the bundle requires more VRAM than the device has.
///
/// Passes (returns nullopt) when:
///   - minimumVramMb is not set in metadata
///   - minimumVramMb is zero or negative
///   - deviceTotalVramMb is zero (device could not be queried — skip disabled)
///   - device has enough VRAM
inline std::optional<std::string> checkVramRequirement(const BundleMetadata& meta,
                                                       std::size_t deviceTotalVramMb)
{
    if(!meta.minimumVramMb || *meta.minimumVramMb <= 0)
    {
        return std::nullopt;
    }
    if(deviceTotalVramMb == 0)
    {
        return std::nullopt;
    }
    if(deviceTotalVramMb < static_cast<std::size_t>(*meta.minimumVramMb))
    {
        return "Bundle requires " + std::to_string(*meta.minimumVramMb)
               + " MB VRAM but device has " + std::to_string(deviceTotalVramMb) + " MB";
    }
    return std::nullopt;
}

/// Skip if the bundle was generated by a GPU reference on a different
/// architecture. CPU-generated golden data is architecture-independent and
/// always passes this check.
///
/// Passes (returns nullopt) when:
///   - referenceExecutor is not set or is "cpu"
///   - gpuArchitecture is not set in metadata
///   - currentArch is empty (device could not be queried — skip disabled)
///   - currentArch contains gpuArchitecture as a substring
inline std::optional<std::string> checkArchCompatibility(const BundleMetadata& meta,
                                                         const std::string& currentArch)
{
    if(!meta.referenceExecutor || *meta.referenceExecutor == "cpu")
    {
        return std::nullopt;
    }
    if(!meta.gpuArchitecture)
    {
        return std::nullopt;
    }
    if(currentArch.empty())
    {
        return std::nullopt;
    }
    if(currentArch.find(*meta.gpuArchitecture) == std::string::npos)
    {
        return "Golden data generated on " + *meta.gpuArchitecture + " but current GPU is "
               + currentArch;
    }
    return std::nullopt;
}

} // namespace hipdnn_test_sdk::utilities

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
