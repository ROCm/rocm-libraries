// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// WinnerCacheFile is layer 3's on-disk record format for the winner cache: the JSON
// envelope one shard line holds, the version line every shard is stamped with and
// checked against, and the path one (engine, arch) shard lives at. It is the only
// header in this directory that depends on all three lower layers at once -- layer 1's
// LineStore/CacheRoot/PathSanitizer, layer 2's GraphContentKey JSON codec, and layer 3's
// own WinnerKey/WinnerRecord/DeviceKey -- which is exactly why it, not layer 1 or 2,
// owns the on-disk envelope (D29).
//
// Every decode here is fail-soft: a missing field, a wrong type, or content that fails
// to reverify returns std::nullopt, never a throw. LineStore's skip-malformed-line
// contract depends on that -- one bad line must cost only itself, not the read of an
// otherwise-good shard.

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

} // namespace detail

/// The bare version string every winner-cache shard is stamped with and checked
/// against -- D24/D30's "one component, named once". Layer 1 (data_sdk) owns the write
/// path (LineStore) and has no other component's version reachable without a new link
/// edge, so its version stamps every shard regardless of which engine or arch it holds.
inline std::string_view winnerCacheVersion()
{
    return HIPDNN_DATA_SDK_VERSION_STRING;
}

/// Where @p engineName's shard for @p gcnArchName lives:
/// `cacheRoot() / "ingestor-winners" / <sdk-version-string> / sanitizeForPath(engineName)
/// / stripArchFeatures(gcnArchName) / "winners.jsonl"`.
///
/// The version-string component is written verbatim, not sanitized: it is
/// build-produced (`MAJOR.MINOR.PATCH.TWEAK`, `HIPDNN_DATA_SDK_VERSION_STRING`), never
/// user- or engine-supplied, and contains no character `sanitizeForPath` would need to
/// touch.
///
/// @return An empty path if `cacheRoot()` cannot resolve a usable cache directory right
///     now (e.g. a filesystem error) -- callers must treat that as "no on-disk cache is
///     available" and fall back to in-memory-only behavior. Never throws.
inline std::filesystem::path winnerCacheShardPath(std::string_view engineName,
                                                  std::string_view gcnArchName)
{
    const auto root = hipdnn_data_sdk::utilities::cacheRoot();
    if(root.empty())
    {
        return {};
    }

    return root / "ingestor-winners" / std::string(winnerCacheVersion())
           / hipdnn_data_sdk::utilities::sanitizeForPath(engineName)
           / std::string(stripArchFeatures(gcnArchName)) / "winners.jsonl";
}

/// Opens (creating if absent) the shard for @p engineName / @p gcnArchName, creating its
/// parent directory tree first -- `openLineStore()` assumes the parent directory already
/// exists, so this is the one place in layer 3 that creates it. Fails soft throughout:
/// an empty `winnerCacheShardPath()` (cacheRoot() unusable) or a directory-creation
/// error both report `LineStoreStatus::OPEN_FAILED` rather than throwing.
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

/// Encodes @p key and @p record as one JSON-Lines record: `key.graph` under layer 2's
/// own codec (`GraphContentKey::toJson()`), `key.device`'s fields folded in directly as
/// plain JSON (`DeviceKey` has no codec of its own), and @p record as an array of ranked
/// entries. `DescriptorId`s use the same UUID text format `DescriptorLoader.hpp` already
/// reads and writes (`hipdnn_flatbuffers_sdk::utilities::formatUuid`/`parseUuid`, reached
/// here via `toString()`) -- no second id format.
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

/// Decodes one line written by encodeWinnerRecordLine(). Fail-soft throughout: a missing
/// or mistyped field, a graph payload `GraphContentKey::fromJson()` declines, or a
/// `DescriptorId` that does not parse as a UUID all return std::nullopt -- matching
/// LineStore's skip-malformed-line contract, never a throw. std::nullopt means "skip
/// this line", not "this shard is corrupt".
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
    catch(const nlohmann::json::exception&)
    {
        return std::nullopt;
    }
    catch(const std::invalid_argument&)
    {
        // parseUuid()'s failure mode for a malformed id.
        return std::nullopt;
    }
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
