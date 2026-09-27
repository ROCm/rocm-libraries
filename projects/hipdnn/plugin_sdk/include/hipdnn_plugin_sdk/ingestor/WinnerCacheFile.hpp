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
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <nlohmann/json.hpp>

#include <hipdnn_data_sdk/utilities/CacheRoot.hpp>
#include <hipdnn_data_sdk/utilities/LineStore.hpp>
#include <hipdnn_data_sdk/utilities/PathSanitizer.hpp>
#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
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
constexpr const char* WINNER_LINE_FORMAT_FIELD = "v";
/// Bump when a record line's fields keep their shape but a meaning underneath one of them
/// changes (decode would still succeed, just on a different interpretation than what was
/// written). A change to GraphContentKey's own shape does not need a bump: it changes the key
/// fromJson() produces, so an old line simply misses lookup instead of being misparsed --
/// self-correcting. This field is the independent per-line stamp: a foreign line (wrong
/// version) appended to an otherwise valid shard is skipped instead of parsed.
constexpr int WINNER_LINE_FORMAT_VERSION = 1;

/// True if @p component is usable verbatim as a path component: a non-empty run of ASCII
/// letters, digits, '_' and '-'.
///
/// Every base target id stripArchFeatures() can produce is of that form -- `gfx942`,
/// `gfx90a`, `gfx1151`, the LLVM generic `gfx9-4-generic` -- and so is every descriptor id,
/// which is UUID text. The check is a whitelist, so a separator, a dot, a colon, a control
/// byte or a non-ASCII byte all fail it, which is what keeps a driver-supplied or
/// author-supplied string inside the cache tree.
inline bool isPlainPathComponent(std::string_view component)
{
    return !component.empty() && std::all_of(component.begin(), component.end(), [](char c) {
        const auto byte = static_cast<unsigned char>(c);
        return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z')
               || (byte >= '0' && byte <= '9') || c == '_' || c == '-';
    });
}

/// A JSON integer within int's range, or nullopt. nlohmann's get<int>() accepts a float and an
/// out-of-range value and static-casts both, which is UB for the out-of-range case.
inline std::optional<int> readBoundedInt(const nlohmann::json& parent, const char* field)
{
    const auto found = parent.find(field);
    if(found == parent.end() || !found->is_number_integer())
    {
        return std::nullopt;
    }
    const auto raw = found->get<int64_t>();
    if(raw < 0 || raw > static_cast<int64_t>(std::numeric_limits<int>::max()))
    {
        return std::nullopt;
    }
    return static_cast<int>(raw);
}

} // namespace detail

/// The version string every winner-cache shard is stamped with and checked against;
/// sourced from data_sdk, which owns the LineStore write path.
inline std::string_view winnerCacheVersion()
{
    return HIPDNN_DATA_SDK_VERSION_STRING;
}

/// Everything a persisted ranking's validity depends on that the `(graph, device)` entry
/// key does not already carry.
///
/// RFC 0019 §9.2: "the UHD identity has to be in the path, because nothing else survives a
/// restart. The moment rankings outlive the process, [the in-process argument] evaporates
/// -- generation counters are process-local, and a restart happily reads entries written by
/// a model that has since been replaced."
///
/// Every member carries a default initializer so a caller can name only what it has --
/// `EngineIdentity{name}` for an engine with no UHD -- without tripping
/// -Wmissing-field-initializers, which this build treats as an error.
struct EngineIdentity
{
    /// `EngineDescriptor::name`. Empty disables the on-disk cache entirely.
    std::string name = {};

    /// `EngineDescriptor::revision`, the descriptor set's authored semantic revision. The
    /// same value `CatalogKey` carries, so an in-memory entry and an on-disk one are
    /// separated by the same event.
    hipdnn_data_sdk::utilities::Version version{};

    /// The catalog-ranking UHD's descriptor id; empty when the engine ships no UHD and
    /// ranks on priority then id.
    std::string uhdId = {};

    /// A content hash over every model this engine can resolve, NOT over the UHD document's
    /// declared version. §9.2: "Hash the content, don't trust the id or a version field. A
    /// regenerated model normally keeps the same UHD id ... and a hand-maintained version can
    /// be forgotten."
    ///
    /// Empty when nothing hashable was declared, which is the case for a native scorer: its
    /// "model" is code compiled into the provider, and the only thing that versions it is the
    /// build, which the data-SDK version component already at the head of the path carries.
    std::string modelHash = {};
};

/// Where @p engine's shard for @p gcnArchName lives:
/// `cacheRoot()/ingestor-winners/<data-sdk version>/<sanitized-engine>/<uhd id>/
///  <engine revision>-<model hash>/<base-arch>/winners.jsonl`.
///
/// The last three components are RFC 0019 §9.2's "directory per heuristic build" with the
/// engine revision folded in, and they are what makes invalidation a directory delete rather
/// than an entry-by-entry staleness check: a new model or a new engine revision writes under
/// a new directory, and the old one is simply unreachable.
///
/// The arch component is the stripped base target id VERBATIM: a user has to be able to
/// find and delete one arch's cache by eye, so `gfx942` must read as `gfx942`. It is
/// sanitizeForPath() would append its unconditional hash suffix and cost exactly that
/// readability, and the arch has nothing to disambiguate, being drawn from a small set of
/// known-good identifiers. An arch that is not a plain component is a driver anomaly:
/// decline the disk cache rather than reshape the string into something that reads like a
/// different arch. The UHD id is treated the same way and for the same reason -- it is UUID
/// text, already a plain component.
///
/// The model hash is truncated to its first 16 hex digits. It is an invalidation token, not
/// an integrity check (the adapters verify the artifact against its full declared checksum
/// when they load it), and 64 bits of it keeps the component short enough that a deep cache
/// root does not push the shard past a filesystem's path limit.
///
/// The device is NOT keyed on the HIP ordinal anywhere in this path or in `WinnerKey`,
/// per §9.2: "Device 0 is a different GPU on a different machine, and can be a different GPU
/// after a reboot." Arch selects the shard; `DeviceKey` carries warpSize and
/// multiProcessorCount inside it.
///
/// @return An empty path if `cacheRoot()` cannot resolve a usable cache directory, or if
///     @p gcnArchName does not strip to a plain component; callers must fall back to
///     in-memory-only behavior. Never throws.
inline std::filesystem::path winnerCacheShardPath(const EngineIdentity& engine,
                                                  std::string_view gcnArchName)
{
    const auto root = hipdnn_data_sdk::utilities::cacheRoot();
    if(root.empty())
    {
        return {};
    }

    const auto arch = stripArchFeatures(gcnArchName);
    if(!detail::isPlainPathComponent(arch))
    {
        return {};
    }

    // "no-uhd" and "unhashed" are distinct directories, not a shared default: an engine that
    // gains a UHD must not inherit the rankings measured while it had none, since those were
    // produced by a different selection path over the same candidates.
    const std::string uhdComponent
        = detail::isPlainPathComponent(engine.uhdId) ? engine.uhdId : "no-uhd";
    const std::string buildComponent
        = engine.version.str() + "-"
          + (detail::isPlainPathComponent(engine.modelHash) ? engine.modelHash.substr(0, 16)
                                                            : "unhashed");

    return root / "ingestor-winners" / std::string(winnerCacheVersion())
           / hipdnn_data_sdk::utilities::sanitizeForPath(engine.name) / uhdComponent
           / buildComponent / std::string(arch) / "winners.jsonl";
}

/// Opens (creating if absent) the shard for @p engine / @p gcnArchName, creating its
/// parent directory tree first. Fails soft: an unusable cache root or a
/// directory-creation error both report `LineStoreStatus::OPEN_FAILED` rather than throwing.
inline std::pair<std::optional<hipdnn_data_sdk::utilities::LineStoreShard>,
                 hipdnn_data_sdk::utilities::LineStoreStatus>
    openWinnerCacheShard(const EngineIdentity& engine, std::string_view gcnArchName)
{
    const auto path = winnerCacheShardPath(engine, gcnArchName);
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
    line[detail::WINNER_LINE_FORMAT_FIELD] = detail::WINNER_LINE_FORMAT_VERSION;
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

        const auto formatField = json.find(detail::WINNER_LINE_FORMAT_FIELD);
        if(formatField == json.end() || !formatField->is_number_integer()
           || formatField->get<int64_t>() != detail::WINNER_LINE_FORMAT_VERSION)
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
        const auto warpSize
            = detail::readBoundedInt(*deviceField, detail::WINNER_LINE_WARP_SIZE_FIELD);
        const auto multiProcessorCount
            = detail::readBoundedInt(*deviceField, detail::WINNER_LINE_MULTI_PROCESSOR_COUNT_FIELD);
        if(!warpSize.has_value() || !multiProcessorCount.has_value())
        {
            return std::nullopt;
        }
        properties.warpSize = *warpSize;
        properties.multiProcessorCount = *multiProcessorCount;

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
