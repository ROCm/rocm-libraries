// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file AutotuneFileWriter.hpp
 * @brief JSON file writer for persisting autotuning results
 *
 * Writes autotune results in heuristic config JSON format, allowing the
 * results to be loaded on subsequent runs via HIPDNN_HEUR_CONFIG_PATH.
 * Supports append/replace semantics and atomic file writes via
 * temp file + rename.
 */

#pragma once

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/autotune/AutotuneTypes.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace hipdnn_frontend
{
namespace autotune
{

/// Serialize a KnobSetting to a JSON object.
inline nlohmann::json knobSettingToJson(const KnobSetting& setting)
{
    nlohmann::json knob;
    knob["knob_id"] = setting.knobId();

    std::visit(
        [&knob](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr(std::is_same_v<T, int64_t>)
            {
                knob["type"] = "int";
                knob["value"] = value;
            }
            else if constexpr(std::is_same_v<T, double>)
            {
                knob["type"] = "double";
                knob["value"] = value;
            }
            else if constexpr(std::is_same_v<T, std::string>)
            {
                knob["type"] = "string";
                knob["value"] = value;
            }
        },
        setting.value());

    return knob;
}

/// Get the lowercase string representation of an AutotuneStrategy (for config file output)
inline std::string strategyToLowerString(AutotuneStrategy strategy)
{
    switch(strategy)
    {
    case AutotuneStrategy::SINGLE_SHOT:
        return "single_shot";
    case AutotuneStrategy::FIXED_AVERAGE:
        return "fixed_average";
    case AutotuneStrategy::RUN_UNTIL_STABLE:
        return "run_until_stable";
    default:
        return "unknown";
    }
}

/// Get the lowercase string representation of a TuneMode (for config file output)
inline std::string tuneModeToLowerString(TuneMode mode)
{
    switch(mode)
    {
    case TuneMode::AUTO:
        return "auto";
    case TuneMode::EXHAUSTIVE:
        return "exhaustive";
    default:
        return "unknown";
    }
}

/// Build a single JSON engine_overrides entry from an AutotuneResult.
///
/// @param result The autotune result to serialize
/// @param opName The operation name for the entry (e.g. "conv_fprop")
/// @param tensorDims Tensor dimensions for the entry (one vector<int64_t> per tensor)
/// @param tensorStrides Tensor strides for the entry (one vector<int64_t> per tensor)
/// @return A nlohmann::json object representing the entry
inline nlohmann::json buildOverrideEntry(const AutotuneResult& result,
                                         const std::string& opName,
                                         const std::vector<std::vector<int64_t>>& tensorDims,
                                         const std::vector<std::vector<int64_t>>& tensorStrides)
{
    nlohmann::json entry;
    entry["op"] = opName;
    entry["engine_name"] = result.engineName;

    // Tensor patterns (dimensions and strides)
    nlohmann::json tensors = nlohmann::json::array();
    for(size_t i = 0; i < tensorDims.size(); ++i)
    {
        nlohmann::json t;
        t["dim"] = tensorDims[i];
        if(i < tensorStrides.size() && !tensorStrides[i].empty())
        {
            t["stride"] = tensorStrides[i];
        }
        tensors.push_back(std::move(t));
    }
    entry["tensors"] = std::move(tensors);

    // Knob settings
    if(!result.knobSettings.empty())
    {
        nlohmann::json knobs = nlohmann::json::array();
        for(const auto& setting : result.knobSettings)
        {
            knobs.push_back(knobSettingToJson(setting));
        }
        entry["knobs"] = std::move(knobs);
    }

    // Autotune metadata
    nlohmann::json metadata;
    metadata["min_time_ms"] = result.minTimeMs;
    metadata["avg_time_ms"] = result.avgTimeMs;
    metadata["stddev_ms"] = result.stddevMs;
    metadata["iterations_run"] = result.iterationsRun;
    metadata["mode"] = tuneModeToLowerString(result.modeUsed);
    metadata["strategy"] = strategyToLowerString(result.strategyUsed);
    metadata["rank"] = result.rank;
    metadata["workspace_size"] = result.workspaceSize;

    // Timestamp in ISO 8601 format, captured at write time
    {
        const auto now = std::chrono::system_clock::now();
        const auto timeT = std::chrono::system_clock::to_time_t(now);
        std::tm utcTm{};
#if defined(_WIN32)
        gmtime_s(&utcTm, &timeT);
#else
        gmtime_r(&timeT, &utcTm);
#endif
        std::ostringstream oss;
        oss << std::put_time(&utcTm, "%Y-%m-%dT%H:%M:%SZ");
        metadata["timestamp"] = oss.str();
    }

    metadata["ran_exhaustive"] = result.ranExhaustive;
    if(result.strategyUsed == AutotuneStrategy::RUN_UNTIL_STABLE)
    {
        metadata["converged"] = result.converged;
    }
    entry["autotune_metadata"] = std::move(metadata);

    return entry;
}

/// Write autotuning results to a JSON file in heuristic config format.
///
/// The file format uses the standard engine_overrides JSON schema:
/// @code{.json}
/// {
///   "engine_overrides": [
///     {
///       "op": "conv_fprop",
///       "engine_name": "MIOPEN_ENGINE",
///       "tensors": [ { "dim": [1, 3, 224, 224] }, { "dim": [64, 3, 7, 7] } ],
///       "knobs": [ { "knob_id": "SPLIT_K", "type": "int", "value": 2 } ],
///       "autotune_metadata": { "min_time_ms": 1.23, "rank": 0 }
///     }
///   ]
/// }
/// @endcode
///
/// @param filePath Output file path
/// @param opName The operation name to use in entries
/// @param results Ranked autotune results (only succeeded entries are written)
/// @param deleteAllExisting When true, starts with an empty file; when false,
///        loads existing entries and replaces matching (op, tensors) entries
/// @param tensorDims Tensor dimensions for the entry
/// @param tensorStrides Tensor strides for the entry
/// @return Error on I/O failure
inline Error writeAutotuneResults(const std::filesystem::path& filePath,
                                  const std::string& opName,
                                  const std::vector<AutotuneResult>& results,
                                  bool deleteAllExisting,
                                  const std::vector<std::vector<int64_t>>& tensorDims,
                                  const std::vector<std::vector<int64_t>>& tensorStrides)
{
    nlohmann::json root;

    // Load existing file content unless we're deleting it all
    if(!deleteAllExisting && std::filesystem::exists(filePath))
    {
        try
        {
            std::ifstream existingFile(filePath);
            if(existingFile.is_open())
            {
                root = nlohmann::json::parse(existingFile);
            }
        }
        catch(const nlohmann::json::exception& e)
        {
            HIPDNN_FE_LOG_ERROR("autotune: existing config file "
                                << filePath
                                << " contains invalid JSON and could not be read: " << e.what()
                                << ". Existing content will be replaced with new results.");
            root = nlohmann::json::object();
        }
    }

    if(!root.contains("engine_overrides") || !root["engine_overrides"].is_array())
    {
        root["engine_overrides"] = nlohmann::json::array();
    }

    // Build the single new entry from the rank-0 winner (the first succeeded
    // result). Only one entry per (op, tensor shape) is ever produced.
    std::optional<nlohmann::json> newEntry;
    for(const auto& result : results)
    {
        if(!result.succeeded)
        {
            continue;
        }

        newEntry = buildOverrideEntry(result, opName, tensorDims, tensorStrides);
        break; // Only write the rank-0 winner
    }

    if(!newEntry.has_value())
    {
        HIPDNN_FE_LOG_WARN("autotune: no successful results to write");
        return {ErrorCode::OK, ""};
    }

    // Remove the pre-existing entry that matches the new entry's (op, tensors)
    // signature, then append the new entry. The knob configuration is
    // unconditionally replaced.
    // Replace-match is exact (operation, tensor shape) only. It does NOT match the
    // reader's -1 wildcard patterns (TensorPattern::matches).
    auto& overrides = root["engine_overrides"];

    if(!overrides.empty())
    {
        overrides.erase(std::remove_if(overrides.begin(),
                                       overrides.end(),
                                       [&](const nlohmann::json& existing) {
                                           return existing.contains("op")
                                                  && existing["op"] == (*newEntry)["op"]
                                                  && existing.contains("tensors")
                                                  && existing["tensors"] == (*newEntry)["tensors"];
                                       }),
                        overrides.end());
    }

    overrides.push_back(*newEntry);

    // Atomic write: write to temp file, then rename.
    // If the process crashes or power is lost between creating the temp file
    // and completing the rename, the temp file (filePath + ".tmp") is left
    // on disk. This is understood and accepted: the alternative (deleting
    // the temp file on failure) risks silently losing the only copy of the
    // data if the rename target was already removed.
    std::filesystem::path tempPath = filePath;
    tempPath += ".tmp";
    {
        std::ofstream outFile(tempPath);
        if(!outFile.is_open())
        {
            return {ErrorCode::INVALID_VALUE,
                    "AutotuneFileWriter: cannot open temp file for writing: " + tempPath.string()};
        }
        outFile << root.dump(2) << '\n';
        outFile.flush();
        if(!outFile.good())
        {
            return {ErrorCode::INVALID_VALUE,
                    "AutotuneFileWriter: write to temp file failed: " + tempPath.string()};
        }
    }

    // Rename temp file to target (atomic on POSIX, best-effort on Windows)
    std::error_code ec;
    std::filesystem::rename(tempPath, filePath, ec);
    if(ec.value() != 0)
    {
        // Fallback: try remove + rename
        std::filesystem::remove(filePath, ec);
        std::filesystem::rename(tempPath, filePath, ec);
        if(ec.value() != 0)
        {
            return {ErrorCode::INVALID_VALUE,
                    "AutotuneFileWriter: failed to rename temp file to " + filePath.string() + ": "
                        + ec.message()};
        }
    }

    HIPDNN_FE_LOG_INFO("autotune: wrote 1 entry to " << filePath);
    return {ErrorCode::OK, ""};
}

} // namespace autotune
} // namespace hipdnn_frontend

#endif // HIPDNN_FRONTEND_SKIP_JSON_LIB
