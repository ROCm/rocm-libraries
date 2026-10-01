// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_test_sdk/utilities/ArchMatch.hpp>

#include <nlohmann/json.hpp>

namespace hipdnn_integration_tests
{

/// How far up RFC 0015's enforcement ladder a bundle/case is checked.
enum class EnforcementLevel
{
    APPLICABILITY, ///< engine must accept the graph
    BUILDABLE, ///< engine must additionally compile a plan
    FULL ///< engine must additionally execute and numeric-verify (default)
};

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
    EnforcementLevel enforcementLevel = EnforcementLevel::FULL;
};

/// Thrown when a metadata object exists but is malformed: not a JSON object,
/// missing/invalid/unsupported `format_version`, an invalid `enforcement_level`,
/// a non-numeric `inputs` key, or (for a .meta.json file) unreadable or not
/// valid JSON. An authoring error,
/// never a "metadata not recorded" case, so it must not degrade to defaults.
class BundleMetadataError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// ---------------------------------------------------------------------------
/// Parse metadata from a JSON object.
///
/// Throws BundleMetadataError when the object is not a JSON object, is missing
/// `format_version`, carries an unsupported format version, has an invalid
/// `enforcement_level`, or has an `inputs` key that is not a tensor UID.
/// `source` names the origin (file path or sweep case) in
/// the error message.
inline BundleMetadata parseBundleMetadataJson(const nlohmann::json& json,
                                              std::string_view source = {})
{
    const std::string where = source.empty() ? std::string("Metadata") : std::string(source);

    if(!json.is_object())
    {
        throw BundleMetadataError(where + " is not a metadata JSON object");
    }

    if(!json.contains("format_version") || !json["format_version"].is_number_integer())
    {
        throw BundleMetadataError(where + " is missing or has an invalid format_version");
    }

    // Compare the JSON value, not a narrowed copy: get<int>() would truncate
    // 4294967297 (2^32 + 1) to 1 and accept it.
    const auto& version = json["format_version"];
    if(version != 1)
    {
        throw BundleMetadataError(where + " has unsupported format_version " + version.dump());
    }

    BundleMetadata meta;
    meta.formatVersion = 1;

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

    // enforcement_level: absent leaves the default Full (set above). Present
    // but not one of the three valid tokens throws (like an invalid
    // format_version) so a typo can't silently flip the level. The
    // RFC §6.2 hard pre-commit error (a claim exists but enforcement_level is
    // missing/invalid) is the claim-bearing cross-check against support.json;
    // it belongs to the downstream enforcement/verifier ticket.
    if(json.contains("enforcement_level"))
    {
        if(!json["enforcement_level"].is_string())
        {
            throw BundleMetadataError(where + " has invalid enforcement_level (not a string)");
        }

        const auto level = json["enforcement_level"].get<std::string>();
        if(level == "applicability")
        {
            meta.enforcementLevel = EnforcementLevel::APPLICABILITY;
        }
        else if(level == "buildable")
        {
            meta.enforcementLevel = EnforcementLevel::BUILDABLE;
        }
        else if(level == "full")
        {
            meta.enforcementLevel = EnforcementLevel::FULL;
        }
        else
        {
            throw BundleMetadataError(where + " has invalid enforcement_level \"" + level + "\"");
        }
    }

    if(json.contains("inputs") && json["inputs"].is_object())
    {
        std::unordered_map<int64_t, nlohmann::json> inputMap;
        for(const auto& [key, val] : json["inputs"].items())
        {
            // Keys are tensor UIDs. A key that is not entirely an integer ("x",
            // "12abc") is an authoring error; dropping it would silently lose
            // that tensor's input spec.
            std::size_t parsed = 0;
            int64_t uid = 0;
            try
            {
                uid = std::stoll(key, &parsed);
            }
            catch(const std::exception&)
            {
                parsed = 0;
            }
            if(parsed == 0 || parsed != key.size())
            {
                std::string message = where + " has non-numeric inputs key \"";
                message += key;
                message += '"';
                throw BundleMetadataError(message);
            }
            inputMap[uid] = val;
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
/// Returns std::nullopt only if the file does not exist (backwards compatible —
/// old bundles simply have no metadata). A file that exists but cannot be
/// opened, is not valid JSON, or is rejected by parseBundleMetadataJson throws
/// BundleMetadataError.
inline std::optional<BundleMetadata> loadBundleMetadata(const std::filesystem::path& bundleJsonPath)
{
    auto path = metaJsonPath(bundleJsonPath);

    if(!std::filesystem::exists(path))
    {
        HIPDNN_SDK_LOG_INFO("No metadata file found at " << path);
        return std::nullopt;
    }

    std::ifstream file(path);
    if(!file)
    {
        throw BundleMetadataError("Could not open metadata file " + path.string());
    }

    nlohmann::json json;
    try
    {
        json = nlohmann::json::parse(file);
    }
    catch(const nlohmann::json::exception& e)
    {
        throw BundleMetadataError("Failed to parse metadata file " + path.string() + ": "
                                  + e.what());
    }
    return parseBundleMetadataJson(json, path.string());
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
