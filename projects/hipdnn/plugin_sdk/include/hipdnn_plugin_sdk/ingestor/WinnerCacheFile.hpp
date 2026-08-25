// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// WinnerCacheFile is the on-disk record format for the winner cache: the JSON envelope
// one shard line holds, the version line every shard is stamped with, and the path an
// (engine, arch) shard lives at.
//
// Every decode here is fail-soft: a missing field, wrong type, or content that fails to
// reverify returns std::nullopt rather than throwing, so LineStore can skip one bad line
// without losing the rest of the shard.

#include <algorithm>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <nlohmann/json.hpp>

#include <hipdnn_data_sdk/utilities/CacheRoot.hpp>
#include <hipdnn_data_sdk/utilities/LineStore.hpp>
#include <hipdnn_data_sdk/utilities/PathSanitizer.hpp>
#include <hipdnn_data_sdk/version.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphContentKey.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceKey.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCache.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

namespace detail
{

constexpr const char* WINNER_LINE_GRAPH_FIELD = "graph";
constexpr const char* WINNER_LINE_DEVICE_FIELD = "device";
constexpr const char* WINNER_LINE_GCN_ARCH_NAME_FIELD = "gcn_arch_name";
constexpr const char* WINNER_LINE_WARP_SIZE_FIELD = "warp_size";
constexpr const char* WINNER_LINE_MULTI_PROCESSOR_COUNT_FIELD = "multi_processor_count";
constexpr const char* WINNER_LINE_ENTRIES_FIELD = "entries";
constexpr const char* WINNER_LINE_KERNEL_ID_FIELD = "kernel_id";
constexpr const char* WINNER_LINE_PACK_ID_FIELD = "pack_id";
constexpr const char* WINNER_LINE_DISPATCH_ID_FIELD = "dispatch_id";
constexpr const char* WINNER_LINE_TIME_MS_FIELD = "time_ms";

/// True if @p arch is usable verbatim as a path component: a non-empty run of ASCII
/// letters, digits, '_' and '-'.
///
/// Every base target id stripArchFeatures() can produce is of that form -- `gfx942`,
/// `gfx90a`, `gfx1151`, the LLVM generic `gfx9-4-generic`. The check is a whitelist, so
/// a separator, a dot, a colon, a control byte or a non-ASCII byte all fail it, which is
/// what keeps a driver-supplied string inside the cache tree.
inline bool isPlainArchComponent(std::string_view arch)
{
    return !arch.empty() && std::all_of(arch.begin(), arch.end(), [](char c) {
        const auto byte = static_cast<unsigned char>(c);
        return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z')
               || (byte >= '0' && byte <= '9') || c == '_' || c == '-';
    });
}

} // namespace detail

/// The version string every winner-cache shard is stamped with and checked against;
/// sourced from data_sdk, which owns the LineStore write path.
inline std::string_view winnerCacheVersion()
{
    return HIPDNN_DATA_SDK_VERSION_STRING;
}

/// Where @p engineName's shard for @p gcnArchName lives:
/// `cacheRoot()/ingestor-winners/<version>/<sanitized-engine>/<base-arch>/winners.jsonl`.
///
/// The arch component is the stripped base target id VERBATIM, which ALMIOPEN-2451's
/// design record requires: a user has to be able to find and delete one arch's cache by
/// eye, so `gfx942` must read as `gfx942`. It is validated rather than sanitized --
/// sanitizeForPath() would append its unconditional hash suffix and cost exactly that
/// readability, and the arch has nothing to disambiguate, being drawn from a small set of
/// known-good identifiers. An arch that is not a plain component is a driver anomaly:
/// decline the disk cache rather than reshape the string into something that reads like a
/// different arch.
///
/// @return An empty path if `cacheRoot()` cannot resolve a usable cache directory, or if
///     @p gcnArchName does not strip to a plain component; callers must fall back to
///     in-memory-only behavior. Never throws.
inline std::filesystem::path winnerCacheShardPath(std::string_view engineName,
                                                  std::string_view gcnArchName)
{
    const auto root = hipdnn_data_sdk::utilities::cacheRoot();
    if(root.empty())
    {
        return {};
    }

    const auto arch = stripArchFeatures(gcnArchName);
    if(!detail::isPlainArchComponent(arch))
    {
        return {};
    }

    return root / "ingestor-winners" / std::string(winnerCacheVersion())
           / hipdnn_data_sdk::utilities::sanitizeForPath(engineName) / std::string(arch)
           / "winners.jsonl";
}

/// Opens (creating if absent) the shard for @p engineName / @p gcnArchName, creating its
/// parent directory tree first. Fails soft: an unusable cache root or a
/// directory-creation error both report `LineStoreStatus::OPEN_FAILED` rather than throwing.
inline std::pair<std::optional<hipdnn_data_sdk::utilities::LineStoreShard>,
                 hipdnn_data_sdk::utilities::LineStoreStatus>
    openWinnerCacheShard(std::string_view engineName, std::string_view gcnArchName)
{
    const auto path = winnerCacheShardPath(engineName, gcnArchName);
    if(path.empty())
    {
        return {std::nullopt, hipdnn_data_sdk::utilities::LineStoreStatus::OPEN_FAILED};
    }

    std::error_code failed;
    std::filesystem::create_directories(path.parent_path(), failed);
    if(failed)
    {
        return {std::nullopt, hipdnn_data_sdk::utilities::LineStoreStatus::OPEN_FAILED};
    }

    return hipdnn_data_sdk::utilities::openLineStore(path, winnerCacheVersion());
}

/// Encodes @p key and @p record as one JSON-Lines record: `key.graph` via
/// `GraphContentKey::toJson()`, `key.device` folded in as plain JSON, and @p record as an
/// array of ranked entries. `DescriptorId`s use the same UUID text format as
/// `DescriptorLoader.hpp` (`formatUuid`/`parseUuid`).
inline std::string encodeWinnerRecordLine(const WinnerKey& key, const WinnerRecord& record)
{
    nlohmann::json device;
    device[detail::WINNER_LINE_GCN_ARCH_NAME_FIELD] = key.device.properties().gcnArchName;
    device[detail::WINNER_LINE_WARP_SIZE_FIELD] = key.device.properties().warpSize;
    device[detail::WINNER_LINE_MULTI_PROCESSOR_COUNT_FIELD]
        = key.device.properties().multiProcessorCount;

    nlohmann::json entries = nlohmann::json::array();
    for(const auto& entry : record)
    {
        nlohmann::json entryJson;
        entryJson[detail::WINNER_LINE_KERNEL_ID_FIELD] = toString(entry.kernelId);
        entryJson[detail::WINNER_LINE_PACK_ID_FIELD] = toString(entry.packId);
        entryJson[detail::WINNER_LINE_DISPATCH_ID_FIELD] = toString(entry.dispatchId);
        entryJson[detail::WINNER_LINE_TIME_MS_FIELD] = entry.timeMs;
        entries.push_back(std::move(entryJson));
    }

    nlohmann::json line;
    line[detail::WINNER_LINE_GRAPH_FIELD] = key.graph.toJson();
    line[detail::WINNER_LINE_DEVICE_FIELD] = std::move(device);
    line[detail::WINNER_LINE_ENTRIES_FIELD] = std::move(entries);
    return line.dump();
}

/// Decodes one line written by `encodeWinnerRecordLine()`. A missing/mistyped field, a
/// graph payload `GraphContentKey::fromJson()` declines, or an unparsable `DescriptorId`
/// all return std::nullopt (never throw), matching LineStore's skip-malformed-line contract.
///
/// The catch is unrestricted because this is `noexcept`: a shard line has no size bound,
/// so parsing one can throw `std::bad_alloc` as readily as a JSON error or `parseUuid()`'s
/// `std::invalid_argument`, and any of them escaping calls `std::terminate`.
inline std::optional<std::pair<WinnerKey, WinnerRecord>>
    decodeWinnerRecordLine(std::string_view line) noexcept
{
    try
    {
        const auto json = nlohmann::json::parse(std::string(line));
        if(!json.is_object())
        {
            return std::nullopt;
        }

        const auto graphField = json.find(detail::WINNER_LINE_GRAPH_FIELD);
        if(graphField == json.end())
        {
            return std::nullopt;
        }
        auto graph
            = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey::fromJson(*graphField);
        if(!graph.has_value())
        {
            return std::nullopt;
        }

        const auto deviceField = json.find(detail::WINNER_LINE_DEVICE_FIELD);
        if(deviceField == json.end() || !deviceField->is_object())
        {
            return std::nullopt;
        }

        DeviceProperties properties;
        properties.gcnArchName
            = deviceField->at(detail::WINNER_LINE_GCN_ARCH_NAME_FIELD).get<std::string>();
        properties.warpSize = deviceField->at(detail::WINNER_LINE_WARP_SIZE_FIELD).get<int>();
        properties.multiProcessorCount
            = deviceField->at(detail::WINNER_LINE_MULTI_PROCESSOR_COUNT_FIELD).get<int>();

        const auto entriesField = json.find(detail::WINNER_LINE_ENTRIES_FIELD);
        if(entriesField == json.end() || !entriesField->is_array())
        {
            return std::nullopt;
        }

        WinnerRecord record;
        record.reserve(entriesField->size());
        for(const auto& entryJson : *entriesField)
        {
            RankedEntry entry;
            entry.kernelId = hipdnn_flatbuffers_sdk::utilities::parseUuid(
                entryJson.at(detail::WINNER_LINE_KERNEL_ID_FIELD).get<std::string>());
            entry.packId = hipdnn_flatbuffers_sdk::utilities::parseUuid(
                entryJson.at(detail::WINNER_LINE_PACK_ID_FIELD).get<std::string>());
            entry.dispatchId = hipdnn_flatbuffers_sdk::utilities::parseUuid(
                entryJson.at(detail::WINNER_LINE_DISPATCH_ID_FIELD).get<std::string>());
            entry.timeMs = entryJson.at(detail::WINNER_LINE_TIME_MS_FIELD).get<double>();
            record.push_back(entry);
        }

        return std::make_pair(WinnerKey{std::move(*graph), DeviceKey{std::move(properties)}},
                              std::move(record));
    }
    catch(...)
    {
        return std::nullopt;
    }
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
