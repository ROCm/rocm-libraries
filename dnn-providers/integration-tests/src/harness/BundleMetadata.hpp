// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_test_sdk/utilities/ArchMatch.hpp>

#include <nlohmann/json.hpp>

#include "harness/input-init/InputFillRecipes.hpp"
#include "harness/input-init/TensorAliases.hpp"

namespace hipdnn_integration_tests
{

// A tensor whose data starts `startOffset` bytes past an aligned address in the
// engine's variant pack.
struct TensorStartOffset
{
    int64_t uid = 0;
    int64_t startOffset = 0;

    bool operator==(const TensorStartOffset& other) const
    {
        return uid == other.uid && startOffset == other.startOffset;
    }
};

// A dense output whose positions past a live length per batch hold no defined value.
// A position is past the live length when its index on the axis is at least the length
// of its batch (index 0). Either `lengthsUid` and `axis` are set, or `likeUid` is set:
// a ragged tensor whose live lengths and sequence axis apply.
struct OutputLiveRows
{
    int64_t uid = 0;
    std::optional<int64_t> axis;
    std::optional<int64_t> lengthsUid;
    std::optional<int64_t> likeUid;

    bool operator==(const OutputLiveRows& other) const
    {
        return uid == other.uid && axis == other.axis && lengthsUid == other.lengthsUid
               && likeUid == other.likeUid;
    }
};

// The alignment of the address that a start offset counts from. A start offset is
// greater than 0 and less than this value.
inline constexpr int64_t K_START_OFFSET_BASE_ALIGNMENT = 256;

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
    /// Inputs that read the memory of another input and have no allocation of their own.
    std::vector<TensorAlias> aliases;
    /// Tensors whose data starts past an aligned address in the engine's variant pack.
    std::vector<TensorStartOffset> startOffsets;
    /// Dense outputs whose positions past a live length are not compared.
    std::vector<OutputLiveRows> liveRows;
    EnforcementLevel enforcementLevel = EnforcementLevel::FULL;

    /// The bundle opts out of the CPU / GPU reference executor as an oracle. A
    /// disabled reference is never created or probed. Disabling both requires
    /// golden data to be declared: nothing else could ever verify the bundle.
    bool noCpuReference = false;
    bool noGpuReference = false;

    /// The bundle is invalid on purpose: the engine under test must reject it, at
    /// ranked-list or plan-build time. Never executed, never compared, and carries
    /// no support claim (see excludedFromSupportClaims()).
    bool expectedRejection = false;
};

/// An expected-rejection bundle is deliberately kept out of support claims, the
/// support matrix and coverage: no engine may claim a graph that every engine must
/// reject. The one definition, read by the harness's claim observation, by
/// registration's claim counters, and by the authoring run (--write-support-claims),
/// so the three cannot drift.
inline bool excludedFromSupportClaims(const BundleMetadata& metadata)
{
    return metadata.expectedRejection;
}

/// Thrown when a metadata object exists but is malformed: not a JSON object,
/// missing/invalid/unsupported `format_version`, an invalid `enforcement_level`,
/// a non-boolean `no_cpu_reference` / `no_gpu_reference` / `expected_rejection`,
/// a non-numeric `inputs` key, an `inputs` entry that is not a valid fill spec, an
/// `aliases` value that parseTensorAliases() rejects, a view whose `inputs` entry sets a
/// fill kind, a malformed `start_offsets` or `live_rows` value, a feature that needs a
/// higher `format_version`, a flag combination that contradicts the bundle's golden data, or
/// (for a .meta.json file) unreadable or not valid JSON.
/// An authoring error, never a "metadata not recorded" case, so it must not degrade
/// to defaults.
class BundleMetadataError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

namespace bundle_metadata_detail
{

// A metadata feature that only format_version 2 metadata may use.
struct FormatVersion2Feature
{
    const char* name;
    bool (*isUsedBy)(const nlohmann::json& metadata);
};

inline bool usesValuesFillKind(const nlohmann::json& metadata)
{
    if(!metadata.contains("inputs") || !metadata["inputs"].is_object())
    {
        return false;
    }
    for(const auto& item : metadata["inputs"].items())
    {
        const auto& spec = item.value();
        if(spec.is_object() && spec.contains("kind") && spec["kind"] == "values")
        {
            return true;
        }
    }
    return false;
}

inline bool usesAliases(const nlohmann::json& metadata)
{
    return metadata.contains("aliases");
}

inline bool usesStartOffsets(const nlohmann::json& metadata)
{
    return metadata.contains("start_offsets");
}

inline bool usesLiveRows(const nlohmann::json& metadata)
{
    return metadata.contains("live_rows");
}

inline constexpr std::array<FormatVersion2Feature, 4> K_FORMAT_VERSION_2_FEATURES = {{
    {"kind values", &usesValuesFillKind},
    {"aliases", &usesAliases},
    {"start_offsets", &usesStartOffsets},
    {"live_rows", &usesLiveRows},
}};

inline std::string describeStartOffset(int64_t uid)
{
    return "start offset of tensor " + std::to_string(uid);
}

inline TensorStartOffset parseStartOffsetEntry(const nlohmann::json& entry,
                                               const std::string& baseWhere)
{
    std::string where = baseWhere;
    const auto uid
        = entry.contains("uid") ? tensor_alias_detail::integerValue(entry.at("uid")) : std::nullopt;
    if(uid.has_value())
    {
        where += " (tensor " + std::to_string(*uid) + ")";
    }

    tensor_alias_detail::rejectUnknownKeys(entry, where, {"uid", "start_offset"});

    TensorStartOffset parsed;
    parsed.uid = tensor_alias_detail::readRequiredInteger(entry, "uid", where);
    parsed.startOffset = tensor_alias_detail::readRequiredInteger(entry, "start_offset", where);
    if(parsed.startOffset <= 0 || parsed.startOffset >= K_START_OFFSET_BASE_ALIGNMENT)
    {
        throw std::invalid_argument(
            describeStartOffset(parsed.uid) + ": start_offset " + std::to_string(parsed.startOffset)
            + " is not in the range 1 to " + std::to_string(K_START_OFFSET_BASE_ALIGNMENT - 1));
    }
    return parsed;
}

// Parses metadata `start_offsets`. Throws std::invalid_argument when the value is not a
// non-empty array of {"uid", "start_offset"} objects with integer values, when a
// start_offset is not in the range 1 to K_START_OFFSET_BASE_ALIGNMENT - 1, or when a uid
// appears twice.
inline std::vector<TensorStartOffset> parseStartOffsets(const nlohmann::json& json)
{
    return tensor_alias_detail::parseUidKeyedArray<TensorStartOffset>(
        json,
        "start_offsets",
        &parseStartOffsetEntry,
        [](const TensorStartOffset& parsed) { return parsed.uid; },
        [](const TensorStartOffset& parsed) {
            return describeStartOffset(parsed.uid) + ": the tensor appears in more than one entry";
        });
}

inline std::string describeLiveRows(int64_t uid)
{
    return "live rows of output " + std::to_string(uid);
}

inline OutputLiveRows parseLiveRowsEntry(const nlohmann::json& entry, const std::string& baseWhere)
{
    std::string where = baseWhere;
    const auto uid
        = entry.contains("uid") ? tensor_alias_detail::integerValue(entry.at("uid")) : std::nullopt;
    if(uid.has_value())
    {
        where += " (output " + std::to_string(*uid) + ")";
    }

    tensor_alias_detail::rejectUnknownKeys(entry, where, {"uid", "axis", "lengths", "like"});

    const auto parsedUid = tensor_alias_detail::readOptionalInteger(entry, "uid", where);
    if(!parsedUid.has_value())
    {
        throw std::invalid_argument(where + R"( lacks "uid")");
    }
    OutputLiveRows parsed;
    parsed.uid = *parsedUid;
    parsed.axis = tensor_alias_detail::readOptionalInteger(entry, "axis", where);
    parsed.lengthsUid = tensor_alias_detail::readOptionalInteger(entry, "lengths", where);
    parsed.likeUid = tensor_alias_detail::readOptionalInteger(entry, "like", where);

    if(parsed.lengthsUid.has_value() == parsed.likeUid.has_value())
    {
        throw std::invalid_argument(where + R"( needs exactly one of "lengths" and "like")");
    }
    if(parsed.lengthsUid.has_value() && !parsed.axis.has_value())
    {
        throw std::invalid_argument(where + R"( has "lengths" and lacks "axis")");
    }
    if(parsed.likeUid.has_value() && parsed.axis.has_value())
    {
        throw std::invalid_argument(
            where + R"( has "like" and "axis"; the axis comes from the ragged tensor)");
    }
    return parsed;
}

// Parses metadata `live_rows`. Throws std::invalid_argument when the value is not a
// non-empty array of objects with an integer "uid" and either integer "axis" and
// "lengths" or an integer "like", when an entry has another key, or when a uid appears
// twice.
inline std::vector<OutputLiveRows> parseLiveRows(const nlohmann::json& json)
{
    return tensor_alias_detail::parseUidKeyedArray<OutputLiveRows>(
        json,
        "live_rows",
        &parseLiveRowsEntry,
        [](const OutputLiveRows& parsed) { return parsed.uid; },
        [](const OutputLiveRows& parsed) {
            return describeLiveRows(parsed.uid) + ": the output appears in more than one entry";
        });
}

} // namespace bundle_metadata_detail

// ---------------------------------------------------------------------------
/// Parse metadata from a JSON object.
///
/// Throws BundleMetadataError when the object is not a JSON object, is missing
/// `format_version`, carries a format version other than 1 or 2, has an invalid
/// `enforcement_level`, has a reference/rejection flag that is not a boolean, has an
/// `inputs` key that is not a tensor UID or an `inputs` entry that is not a valid fill
/// spec, has an `aliases` value that parseTensorAliases() rejects or a view whose
/// `inputs` entry sets a fill kind, has a malformed `start_offsets` or `live_rows` value,
/// or uses a version-2 feature in version 1 metadata.
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
    if(version != 1 && version != 2)
    {
        throw BundleMetadataError(where + " has unsupported format_version " + version.dump());
    }

    BundleMetadata meta;
    meta.formatVersion = version.get<int>();

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

    // The three flags are strict booleans. Absent means false; a string, a number
    // or a null is an authoring error, because "yes" quietly reading as false would
    // leave the bundle verified by the reference it meant to switch off.
    auto readFlag = [&](const char* key) -> bool {
        if(!json.contains(key))
        {
            return false;
        }
        if(!json[key].is_boolean())
        {
            throw BundleMetadataError(where + " has invalid " + key + " (not a boolean)");
        }
        return json[key].get<bool>();
    };
    meta.noCpuReference = readFlag("no_cpu_reference");
    meta.noGpuReference = readFlag("no_gpu_reference");
    meta.expectedRejection = readFlag("expected_rejection");

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
            try
            {
                parseInputFillSpec(val);
            }
            catch(const std::invalid_argument& e)
            {
                std::string message = where;
                message += " input ";
                message += key;
                message += ": ";
                message += e.what();
                throw BundleMetadataError(message);
            }
            inputMap[uid] = val;
        }
        meta.inputs = std::move(inputMap);
    }

    // Version-1 metadata with `aliases` fails the version check below, whatever the value.
    if(json.contains("aliases") && meta.formatVersion >= 2)
    {
        try
        {
            meta.aliases = parseTensorAliases(json.at("aliases"));
        }
        catch(const std::invalid_argument& e)
        {
            throw BundleMetadataError(where + ": " + e.what());
        }
    }

    // Version-1 metadata with `start_offsets` fails the version check below, whatever the
    // value.
    if(json.contains("start_offsets") && meta.formatVersion >= 2)
    {
        try
        {
            meta.startOffsets = bundle_metadata_detail::parseStartOffsets(json.at("start_offsets"));
        }
        catch(const std::invalid_argument& e)
        {
            throw BundleMetadataError(where + ": " + e.what());
        }
    }

    // Version-1 metadata with `live_rows` fails the version check below, whatever the value.
    if(json.contains("live_rows") && meta.formatVersion >= 2)
    {
        try
        {
            meta.liveRows = bundle_metadata_detail::parseLiveRows(json.at("live_rows"));
        }
        catch(const std::invalid_argument& e)
        {
            throw BundleMetadataError(where + ": " + e.what());
        }
    }

    // A seed-only entry for a view is allowed and has no effect.
    for(const auto& alias : meta.aliases)
    {
        if(!meta.inputs.has_value())
        {
            break;
        }
        const auto entry = meta.inputs->find(alias.uid);
        if(entry != meta.inputs->end() && entry->second.contains("kind"))
        {
            std::string message = where;
            message.append(": ")
                .append(tensor_alias_detail::describe(alias))
                .append(": the view's inputs entry sets a fill kind, and a view is never filled");
            throw BundleMetadataError(message);
        }
    }

    if(meta.formatVersion < 2)
    {
        for(const auto& feature : bundle_metadata_detail::K_FORMAT_VERSION_2_FEATURES)
        {
            if(feature.isUsedBy(json))
            {
                throw BundleMetadataError(where + ": " + feature.name + " needs format_version 2");
            }
        }
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
