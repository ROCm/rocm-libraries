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

#include <hipdnn_data_sdk/detail/AutotuneConfigNames.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/autotune/AutotuneTypes.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hipdnn_frontend
{
namespace autotune
{
namespace detail
{
namespace config_json = hipdnn_data_sdk::detail::autotune_config::json;
namespace config_version = hipdnn_data_sdk::detail::autotune_config::version;

using Criteria = std::vector<std::pair<std::string, int64_t>>;

inline nlohmann::json criteriaToJson(const Criteria& criteria)
{
    nlohmann::json result = nlohmann::json::object();
    for(const auto& [key, value] : criteria)
    {
        result[key] = value;
    }
    return result;
}

inline nlohmann::json criteriaOrEmpty(const nlohmann::json& entry)
{
    if(entry.contains(config_json::CRITERIA) && entry[config_json::CRITERIA].is_object())
    {
        return entry[config_json::CRITERIA];
    }
    return nlohmann::json::object();
}

enum class TensorIdValidationResult
{
    HIPDNN_AUTOTUNE_TENSOR_IDS_VALID,
    HIPDNN_AUTOTUNE_TENSOR_ID_MISSING,
    HIPDNN_AUTOTUNE_TENSOR_ID_DUPLICATE
};

inline TensorIdValidationResult validateTensorIds(size_t tensorCount,
                                                  const std::vector<std::string>& tensorIds)
{
    if(tensorCount == 0 || tensorIds.size() < tensorCount)
    {
        return TensorIdValidationResult::HIPDNN_AUTOTUNE_TENSOR_ID_MISSING;
    }

    std::unordered_set<std::string_view> seen;
    seen.reserve(tensorCount);
    for(size_t i = 0; i < tensorCount; ++i)
    {
        const auto& tensorId = tensorIds[i];
        if(tensorId.empty())
        {
            return TensorIdValidationResult::HIPDNN_AUTOTUNE_TENSOR_ID_MISSING;
        }
        if(!seen.emplace(tensorId.data(), tensorId.size()).second)
        {
            return TensorIdValidationResult::HIPDNN_AUTOTUNE_TENSOR_ID_DUPLICATE;
        }
    }

    return TensorIdValidationResult::HIPDNN_AUTOTUNE_TENSOR_IDS_VALID;
}

inline std::optional<std::string_view> tensorIdOf(const nlohmann::json& tensor)
{
    if(!tensor.contains(config_json::TENSOR_ID) || !tensor[config_json::TENSOR_ID].is_string())
    {
        return std::nullopt;
    }

    const auto& tensorId = tensor[config_json::TENSOR_ID].get_ref<const std::string&>();
    return std::string_view(tensorId.data(), tensorId.size());
}

inline nlohmann::json tensorEntryWithoutId(nlohmann::json entry)
{
    entry.erase(config_json::TENSOR_ID);
    return entry;
}

inline bool tensorsMatchByIdIgnoringOrder(const nlohmann::json& existing,
                                          const nlohmann::json& replacement)
{
    std::unordered_map<std::string_view, const nlohmann::json*> existingById;
    existingById.reserve(existing.size());
    for(const auto& existingTensor : existing)
    {
        const auto existingId = tensorIdOf(existingTensor);
        if(!existingId.has_value() || !existingById.emplace(*existingId, &existingTensor).second)
        {
            return false;
        }
    }

    return std::all_of(
        replacement.begin(), replacement.end(), [&](const nlohmann::json& replacementTensor) {
            const auto replacementId = tensorIdOf(replacementTensor);
            if(!replacementId.has_value())
            {
                return false;
            }

            const auto existingIt = existingById.find(*replacementId);
            return existingIt != existingById.end()
                   && tensorEntryWithoutId(*existingIt->second)
                          == tensorEntryWithoutId(replacementTensor);
        });
}

inline bool tensorSignaturesMatch(const nlohmann::json& existing, const nlohmann::json& replacement)
{
    return existing.is_array() && replacement.is_array() && existing.size() == replacement.size()
           && tensorsMatchByIdIgnoringOrder(existing, replacement);
}

// Build a single JSON engine_overrides entry from an AutotuneResult.
// tensorDims/tensorStrides hold one vector<int64_t> per tensor. opName is the
// operation name for the entry (e.g. "conv_fprop"). Returns a nlohmann::json
// object representing the entry.
inline nlohmann::json buildOverrideEntry(const AutotuneResult& result,
                                         const std::string& opName,
                                         const std::vector<std::vector<int64_t>>& tensorDims,
                                         const std::vector<std::vector<int64_t>>& tensorStrides,
                                         const Criteria& criteria = {},
                                         const std::vector<std::string>& tensorIds = {})
{
    nlohmann::json entry;
    entry[config_json::OP] = opName;
    if(!criteria.empty())
    {
        entry[config_json::CRITERIA] = criteriaToJson(criteria);
    }
    entry[config_json::ENGINE_NAME] = result.engineName;

    // Tensor patterns (dimensions and strides)
    nlohmann::json tensors = nlohmann::json::array();
    for(size_t i = 0; i < tensorDims.size(); ++i)
    {
        nlohmann::json t;
        t[config_json::DIM] = tensorDims[i];
        if(i < tensorIds.size() && !tensorIds[i].empty())
        {
            t[config_json::TENSOR_ID] = tensorIds[i];
        }
        if(i < tensorStrides.size() && !tensorStrides[i].empty())
        {
            t[config_json::STRIDE] = tensorStrides[i];
        }
        tensors.push_back(std::move(t));
    }
    entry[config_json::TENSORS] = std::move(tensors);

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

    metadata["supports_exhaustive"] = result.supportsExhaustive;
    metadata["ran_exhaustive"] = result.ranExhaustive;
    metadata["exhaustive_not_run_reason"] = result.exhaustiveNotRunReason;
    if(result.strategyUsed == AutotuneStrategy::RUN_UNTIL_STABLE)
    {
        metadata["converged"] = result.converged;
    }

    // Knob settings (informational; nested under autotune_metadata, omitted when empty)
    if(!result.knobSettings.empty())
    {
        nlohmann::json knobs = nlohmann::json::array();
        for(const auto& setting : result.knobSettings)
        {
            // KnobSetting -> JSON via the ADL-found to_json
            knobs.push_back(setting);
        }
        metadata["knobs"] = std::move(knobs);
    }

    entry["autotune_metadata"] = std::move(metadata);

    return entry;
}

// Write autotuning results to a JSON file in heuristic config format.
//
// The file format uses the versioned engine_overrides JSON schema. A single
// rank-0 winner entry looks like:
// {
//   "version": 2,
//   "engine_overrides": [
//     {
//       "op": "conv_fprop",
//       "engine_name": "MIOPEN_ENGINE",
//       "criteria": { "pointwise_mode": 1 },
//       "tensors": [
//         { "tensor_id": "X", "dim": [1, 3, 224, 224], "stride": [150528, 50176, 224, 1] },
//         { "tensor_id": "W", "dim": [64, 3, 7, 7], "stride": [147, 49, 7, 1] }
//       ],
//       "autotune_metadata": {
//         "rank": 0,
//         "min_time_ms": 1.23,
//         "avg_time_ms": 1.25,
//         "stddev_ms": 0.02,
//         "workspace_size": 16777216,
//         "mode": "exhaustive",
//         "supports_exhaustive": true,
//         "ran_exhaustive": true,
//         "exhaustive_not_run_reason": "",
//         "strategy": "run_until_stable",
//         "iterations_run": 37,
//         "converged": true,
//         "timestamp": "2026-04-21T10:30:00Z",
//         "knobs": [ { "knob_id": "SPLIT_K", "type": "int", "value": 2 } ]
//       }
//     }
//   ]
// }
//
// op is the canonical core-operation string (conv_fprop, conv_dgrad, matmul,
// sdpa_fwd, batchnorm_training, layernorm, pointwise, etc.). criteria is
// present only when the graph supplies discriminating criteria. Per tensor,
// tensor_id is required for the v2 named-id format and stride is present when
// strides are supplied. In autotune_metadata, converged is present only for
// the run_until_stable strategy, and knobs is omitted entirely for
// default-knob entries.
//
// Writes a single entry: the rank-0 winner (the first succeeded result in the
// rank-ordered input). If no result succeeded, nothing is written and OK is
// returned. When deleteAllExisting is true the file starts empty; when false,
// existing entries are loaded and the one matching (op, criteria, tensors) is
// replaced. Returns an Error on I/O failure.
inline Error writeAutotuneResults(const std::filesystem::path& filePath,
                                  const std::string& opName,
                                  const std::vector<AutotuneResult>& results,
                                  bool deleteAllExisting,
                                  const std::vector<std::vector<int64_t>>& tensorDims,
                                  const std::vector<std::vector<int64_t>>& tensorStrides,
                                  const Criteria& criteria = {},
                                  const std::vector<std::string>& tensorIds = {})
{
    try
    {
        nlohmann::json root;
        bool loadedExistingConfig = false;

        // Load existing file content unless we're overwriting it all.
        if(!deleteAllExisting && std::filesystem::exists(filePath))
        {
            try
            {
                std::ifstream existingFile(filePath);
                if(existingFile.is_open())
                {
                    root = nlohmann::json::parse(existingFile);
                    loadedExistingConfig = true;
                }
            }
            catch(const nlohmann::json::exception& e)
            {
                // The existing file is corrupt. Attempt to move aside.
                const auto epochSeconds = std::chrono::duration_cast<std::chrono::seconds>(
                                              std::chrono::system_clock::now().time_since_epoch())
                                              .count();
                std::filesystem::path corruptPath = filePath;
                corruptPath += ".corrupt-" + std::to_string(epochSeconds);
                std::error_code renameEc;
                std::filesystem::rename(filePath, corruptPath, renameEc);
                if(renameEc)
                {
                    HIPDNN_FE_LOG_ERROR("autotune: existing config file "
                                        << filePath << " contains invalid JSON (" << e.what()
                                        << ") and could not be moved aside (" << renameEc.message()
                                        << "); overwriting it with new results");
                }
                else
                {
                    HIPDNN_FE_LOG_WARN(
                        "autotune: existing config file "
                        << filePath << " contains invalid JSON and could not be read: " << e.what()
                        << ". Moved aside to " << corruptPath << "; writing fresh results.");
                }
                root = nlohmann::json::object();
            }
        }

        if(loadedExistingConfig)
        {
            if(!root.is_object())
            {
                return {ErrorCode::INVALID_VALUE,
                        "AutotuneFileWriter: existing config file is not a versioned object"};
            }
            if(!root.contains(config_json::VERSION))
            {
                return {ErrorCode::INVALID_VALUE,
                        "AutotuneFileWriter: refusing to update legacy autotune config file"};
            }
            const auto& existingConfigVersion = root.at(config_json::VERSION);
            if(!existingConfigVersion.is_number_integer())
            {
                return {ErrorCode::INVALID_VALUE,
                        "AutotuneFileWriter: existing config version is not an integer"};
            }
            if(existingConfigVersion.get<int64_t>() != config_version::CURRENT)
            {
                return {ErrorCode::INVALID_VALUE,
                        "AutotuneFileWriter: refusing to update non-current autotune config file"};
            }
        }
        else
        {
            root = nlohmann::json::object();
        }

        if(!root.contains(config_json::ENGINE_OVERRIDES)
           || !root[config_json::ENGINE_OVERRIDES].is_array())
        {
            root[config_json::ENGINE_OVERRIDES] = nlohmann::json::array();
        }

        // Build the single new entry from the rank-0 winner (the first succeeded
        // result). Only one entry per (op, tensor shape) is ever produced.
        const auto rank0Result
            = std::find_if(results.begin(), results.end(), [](const AutotuneResult& result) {
                  return result.succeeded;
              });
        if(rank0Result == results.end())
        {
            HIPDNN_FE_LOG_WARN("autotune: no successful results to write");
            return {ErrorCode::OK, ""};
        }

        switch(validateTensorIds(tensorDims.size(), tensorIds))
        {
        case TensorIdValidationResult::HIPDNN_AUTOTUNE_TENSOR_IDS_VALID:
            break;
        case TensorIdValidationResult::HIPDNN_AUTOTUNE_TENSOR_ID_MISSING:
            return {ErrorCode::INVALID_VALUE,
                    "AutotuneFileWriter: tensor IDs are required for versioned config files"};
        case TensorIdValidationResult::HIPDNN_AUTOTUNE_TENSOR_ID_DUPLICATE:
            return {ErrorCode::INVALID_VALUE,
                    "AutotuneFileWriter: tensor IDs must be unique for versioned config files"};
        default:
            return {ErrorCode::INVALID_VALUE,
                    "AutotuneFileWriter: unknown tensor ID validation result"};
        }

        const auto newEntry = buildOverrideEntry(
            *rank0Result, opName, tensorDims, tensorStrides, criteria, tensorIds);

        root[config_json::VERSION] = config_version::CURRENT;

        // Remove the pre-existing entry that matches the new entry's (op, tensors)
        // signature, then append the new entry. The knob configuration is
        // unconditionally replaced.
        // Replace-match is exact (operation, tensor shape) only. It does NOT match the
        // reader's -1 wildcard patterns (TensorPattern::matches).
        auto& overrides = root[config_json::ENGINE_OVERRIDES];

        if(!overrides.empty())
        {
            overrides.erase(
                std::remove_if(overrides.begin(),
                               overrides.end(),
                               [&](const nlohmann::json& existing) {
                                   return existing.contains(config_json::OP)
                                          && existing[config_json::OP] == newEntry[config_json::OP]
                                          && criteriaOrEmpty(existing) == criteriaOrEmpty(newEntry)
                                          && existing.contains(config_json::TENSORS)
                                          && tensorSignaturesMatch(existing[config_json::TENSORS],
                                                                   newEntry[config_json::TENSORS]);
                               }),
                overrides.end());
        }

        overrides.push_back(newEntry);

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
                        "AutotuneFileWriter: cannot open temp file for writing: "
                            + tempPath.string()};
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
            // Fallback: remove the existing target, then retry the rename. If the
            // remove itself fails there is no point retrying - report it.
            std::filesystem::remove(filePath, ec);
            if(ec.value() != 0)
            {
                return {ErrorCode::INVALID_VALUE,
                        "AutotuneFileWriter: failed to remove existing file " + filePath.string()
                            + " before rename: " + ec.message()};
            }
            std::filesystem::rename(tempPath, filePath, ec);
            if(ec.value() != 0)
            {
                return {ErrorCode::INVALID_VALUE,
                        "AutotuneFileWriter: failed to rename temp file to " + filePath.string()
                            + ": " + ec.message()};
            }
        }

        HIPDNN_FE_LOG_INFO("autotune: wrote 1 entry to " << filePath);
        return {ErrorCode::OK, ""};
    }
    catch(const std::exception& e)
    {
        return {ErrorCode::INVALID_VALUE, e.what()};
    }
}

} // namespace detail

} // namespace autotune
} // namespace hipdnn_frontend

#endif // HIPDNN_FRONTEND_SKIP_JSON_LIB
