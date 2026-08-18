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
#include <string_view>
#include <unordered_map>

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_test_sdk/utilities/ArchMatch.hpp>

#include <nlohmann/json.hpp>

namespace hipdnn_integration_tests
{

/// Metadata read from a {Name}.meta.json companion file alongside a golden
/// reference bundle ({Name}.json + {Name}.tensor{uid}.bin).
///
/// All fields are at the top level of the JSON object (RFC 0011 §4.1).
/// Every field except `format_version` is optional. A missing field means
/// "not recorded" — the system must behave correctly without it.
struct BundleMetadata
{
    int formatVersion = 1;

    std::optional<std::string> generator;
    std::optional<std::string> generatorVersion;
    std::optional<std::string> generatedAt;
    std::optional<std::string> gpuArchitecture;
    std::optional<std::string> rocmVersion;
    std::optional<std::string> referenceSource;
    std::optional<std::string> referenceSourceHash;
    std::optional<std::string> referenceStrategy;
    std::optional<std::string> operation;
    std::optional<std::string> generationCommand;
    std::optional<std::string> notes;
    std::optional<int64_t> seed;
    std::optional<int64_t> minimumVramMb;
    std::optional<std::unordered_map<int64_t, nlohmann::json>> inputs;
};

// ---------------------------------------------------------------------------
/// Parse metadata from a JSON object.
///
/// Returns std::nullopt when the object is missing required fields or carries an
/// unsupported format version. Never throws.
inline std::optional<BundleMetadata> parseBundleMetadataJson(const nlohmann::json& json,
                                                             std::string_view source = {})
{
    if(!json.is_object())
    {
        if(source.empty())
        {
            HIPDNN_SDK_LOG_WARN("Metadata JSON is not an object");
        }
        else
        {
            HIPDNN_SDK_LOG_WARN(source << " is not a metadata JSON object");
        }
        return std::nullopt;
    }

    if(!json.contains("format_version") || !json["format_version"].is_number_integer())
    {
        if(source.empty())
        {
            HIPDNN_SDK_LOG_WARN("Metadata missing or invalid format_version");
        }
        else
        {
            HIPDNN_SDK_LOG_WARN(source << " missing or invalid format_version");
        }
        return std::nullopt;
    }

    const int version = json["format_version"].get<int>();
    if(version != 1)
    {
        if(source.empty())
        {
            HIPDNN_SDK_LOG_WARN("Metadata has unsupported format_version " << version);
        }
        else
        {
            HIPDNN_SDK_LOG_WARN(source << " has unsupported format_version " << version);
        }
        return std::nullopt;
    }

    BundleMetadata meta;
    meta.formatVersion = version;

    auto readString = [&](const char* key) -> std::optional<std::string> {
        if(json.contains(key) && json[key].is_string())
        {
            return json[key].get<std::string>();
        }
        return std::nullopt;
    };

    auto readInt64 = [&](const char* key) -> std::optional<int64_t> {
        if(json.contains(key) && json[key].is_number_integer())
        {
            return json[key].get<int64_t>();
        }
        return std::nullopt;
    };

    meta.generator = readString("generator");
    meta.generatorVersion = readString("generator_version");
    meta.generatedAt = readString("generated_at");
    meta.gpuArchitecture = readString("gpu_architecture");
    meta.rocmVersion = readString("rocm_version");
    meta.referenceSource = readString("reference_source");
    meta.referenceSourceHash = readString("reference_source_hash");
    meta.referenceStrategy = readString("reference_strategy");
    meta.operation = readString("operation");
    meta.generationCommand = readString("generation_command");
    meta.notes = readString("notes");
    meta.seed = readInt64("seed");
    meta.minimumVramMb = readInt64("minimum_vram_mb");

    if(json.contains("inputs") && json["inputs"].is_object())
    {
        std::unordered_map<int64_t, nlohmann::json> inputMap;
        for(const auto& [key, val] : json["inputs"].items())
        {
            try
            {
                inputMap[std::stoll(key)] = val;
            }
            catch(const std::exception&)
            {
                HIPDNN_SDK_LOG_WARN("Skipping non-numeric inputs key \""
                                    << key << "\" in " << (source.empty() ? "metadata" : source));
                continue;
            }
        }
        meta.inputs = std::move(inputMap);
    }

    return meta;
}
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
inline std::optional<BundleMetadata> loadBundleMetadata(const std::filesystem::path& bundleJsonPath)
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
        return parseBundleMetadataJson(json, path.string());
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
        return "Bundle requires " + std::to_string(*meta.minimumVramMb) + " MB VRAM but device has "
               + std::to_string(deviceTotalVramMb) + " MB";
    }
    return std::nullopt;
}

/// Skip if the bundle's golden data is tied to a different GPU architecture.
///
/// Architecture dependence is determined solely by the `gpu_architecture`
/// field in meta.json. When present (e.g. "gfx942"), the data was generated
/// by an arch-specific tool (AITER, a GPU reference executor) whose numerical
/// output varies by architecture. When absent, the data is assumed portable
/// across all ASICs (e.g. generated by PyTorch, a CPU reference executor).
///
/// `reference_source` is NOT consulted — it is informational only.
///
/// Passes (returns nullopt) when:
///   - gpuArchitecture is not set or is empty in metadata (portable data)
///   - currentArch is empty (device could not be queried — skip disabled)
///   - currentArch matches gpuArchitecture at the base level (before ':' suffix)
inline std::optional<std::string> checkArchCompatibility(const BundleMetadata& meta,
                                                         const std::string& currentArch)
{
    if(!meta.gpuArchitecture || meta.gpuArchitecture->empty())
    {
        return std::nullopt;
    }
    if(currentArch.empty())
    {
        return std::nullopt;
    }
    // Strict match: golden data is arch-locked, so data generated on gfx942
    // must not run on gfx940. e.g. metadata "gfx942" matches device
    // "gfx942:sramecc+:xnack-" but not "gfx940".
    const auto& metaArch = *meta.gpuArchitecture;
    if(!hipdnn_test_sdk::utilities::archMatches(
           currentArch, metaArch, hipdnn_test_sdk::utilities::ArchMatchMode::PREFIX))
    {
        return "Golden data generated on " + *meta.gpuArchitecture + " but current GPU is "
               + currentArch;
    }
    return std::nullopt;
}

} // namespace hipdnn_integration_tests

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
