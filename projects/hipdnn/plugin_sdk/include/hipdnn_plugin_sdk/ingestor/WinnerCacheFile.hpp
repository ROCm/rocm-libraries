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
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
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
constexpr const char* WINNER_LINE_TOTAL_GLOBAL_MEM_FIELD = "total_global_mem";
constexpr const char* WINNER_LINE_MEMORY_BUS_WIDTH_FIELD = "memory_bus_width";
constexpr const char* WINNER_LINE_MEMORY_CLOCK_RATE_FIELD = "memory_clock_rate";
constexpr const char* WINNER_LINE_SHARED_MEM_PER_BLOCK_FIELD = "shared_mem_per_block";
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
///
/// Version 2 added the memory fields DeviceKey compares; version-1 lines are skipped
/// rather than decoded with those fields zeroed.
constexpr int WINNER_LINE_FORMAT_VERSION = 2;

/// True if @p component is usable verbatim as a path component: a non-empty run of ASCII
/// letters, digits, '_' and '-'.
///
/// Every base target id stripArchFeatures() can produce is of that form -- `gfx942`,
/// `gfx90a`, `gfx1151`, `gfx9-4-generic` -- as is every UUID-text descriptor id. Being a
/// whitelist keeps driver- or author-supplied strings inside the cache tree.
inline bool isPlainPathComponent(std::string_view component)
{
    return !component.empty() && std::all_of(component.begin(), component.end(), [](char c) {
        const auto byte = static_cast<unsigned char>(c);
        return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z')
               || (byte >= '0' && byte <= '9') || c == '_' || c == '-';
    });
}

/// A JSON integer within [0, max of @p T], or nullopt. nlohmann's get<T>() would silently
/// cast floats, negatives and out-of-range values.
template <typename T>
inline std::optional<T> readNonNegative(const nlohmann::json& parent, const char* field)
{
    static_assert(std::is_integral_v<T>);
    const auto found = parent.find(field);
    if(found == parent.end() || !found->is_number_integer())
    {
        return std::nullopt;
    }
    // A non-negative JSON integer is stored unsigned or signed depending on how it was
    // produced; reject a negative one before widening to uint64_t.
    if(found->is_number_unsigned())
    {
        const auto raw = found->get<uint64_t>();
        if(raw > static_cast<uint64_t>(std::numeric_limits<T>::max()))
        {
            return std::nullopt;
        }
        return static_cast<T>(raw);
    }
    const auto raw = found->get<int64_t>();
    if(raw < 0 || static_cast<uint64_t>(raw) > static_cast<uint64_t>(std::numeric_limits<T>::max()))
    {
        return std::nullopt;
    }
    return static_cast<T>(raw);
}

} // namespace detail

/// The version string every winner-cache shard is stamped with and checked against;
/// sourced from data_sdk, which owns the LineStore write path.
inline std::string_view winnerCacheVersion()
{
    return HIPDNN_DATA_SDK_VERSION_STRING;
}

/// Everything a persisted ranking's validity depends on that the `(graph, device)` entry
/// key does not carry (RFC 0019 §9.2: only what is in the path survives a restart).
/// Default initializers let callers name only what they have, e.g. `EngineIdentity{name}`,
/// without -Wmissing-field-initializers.
struct EngineIdentity
{
    // NOLINTBEGIN(readability-redundant-member-init) - the initializers are load-bearing, see above
    /// `EngineDescriptor::name`. Empty disables the on-disk cache entirely.
    std::string name = {};

    /// `EngineDescriptor::revision`; the same value `CatalogKey` carries.
    hipdnn_data_sdk::utilities::Version version{};

    /// The catalog-ranking UHD's descriptor id; empty when the engine ships no UHD and
    /// ranks on priority then id.
    std::string uhdId = {};

    /// Content hash over every model this engine can resolve, not a declared version
    /// (RFC 0019 §9.2). Empty when the engine ships no heuristic; a native scorer's model is
    /// code, versioned by the data-SDK component at the head of the path.
    std::string modelHash = {};

    /// False when a resolvable model names an artifact with no digest (none declared, no
    /// bytes at load). @ref modelHash cannot version it, so the on-disk cache is declined.
    bool contentIdentified = true;
    // NOLINTEND(readability-redundant-member-init)
};

/// Where @p engine's shard for @p gcnArchName lives:
/// `cacheRoot()/ingestor-winners/<data-sdk version>/<sanitized-engine>/<uhd id>/
///  <engine revision>-<model hash>/<base-arch>/winners.jsonl`.
///
/// The last three components are RFC 0019 §9.2's "directory per heuristic build", so a
/// new model or engine revision invalidates by writing to a new directory.
///
/// The arch component is the stripped base target id VERBATIM: a user has to be able to
/// find and delete one arch's cache by eye, so `gfx942` must read as `gfx942`. It is
/// sanitizeForPath() would append its unconditional hash suffix and cost exactly that
/// readability, and the arch has nothing to disambiguate, being drawn from a small set of
/// known-good identifiers. An arch that is not a plain component is a driver anomaly:
/// decline the disk cache rather than reshape the string into something that reads like a
/// different arch. The UHD id (UUID text) is checked the same way.
///
/// The model hash is truncated to 16 hex digits: an invalidation token, not an integrity
/// check, kept short to stay under filesystem path limits.
///
/// The device is never keyed on the HIP ordinal (RFC 0019 §9.2): arch selects the
/// shard, and `DeviceKey` in each record carries the rest of the device identity.
///
/// @return An empty path if `cacheRoot()` cannot resolve a usable cache directory, if
///     @p gcnArchName does not strip to a plain component, or if @p engine has no content
///     identity (EngineIdentity::contentIdentified); callers must fall back to
///     in-memory-only behavior. Never throws.
inline std::filesystem::path winnerCacheShardPath(const EngineIdentity& engine,
                                                  std::string_view gcnArchName)
{
    if(!engine.contentIdentified)
    {
        return {};
    }
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

    // Distinct from "unhashed": an engine that gains a UHD must not inherit rankings
    // measured by a different selection path.
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
/// `GraphContentKey::toJson()`, every field of `key.device` as plain JSON, and @p record as an
/// array of ranked entries. `DescriptorId`s use the same UUID text format as
/// `DescriptorLoader.hpp` (`formatUuid`/`parseUuid`).
inline std::string encodeWinnerRecordLine(const WinnerKey& key, const WinnerRecord& record)
{
    // Structured binding so growing DeviceProperties stops this compiling until the codec
    // (and WINNER_LINE_FORMAT_VERSION) persist the new field; DeviceKey compares them all.
    const auto& [gcnArchName,
                 warpSize,
                 multiProcessorCount,
                 totalGlobalMem,
                 memoryBusWidth,
                 memoryClockRate,
                 sharedMemPerBlock]
        = key.device.properties();
    nlohmann::json device;
    device[detail::WINNER_LINE_GCN_ARCH_NAME_FIELD] = gcnArchName;
    device[detail::WINNER_LINE_WARP_SIZE_FIELD] = warpSize;
    device[detail::WINNER_LINE_MULTI_PROCESSOR_COUNT_FIELD] = multiProcessorCount;
    device[detail::WINNER_LINE_TOTAL_GLOBAL_MEM_FIELD] = totalGlobalMem;
    device[detail::WINNER_LINE_MEMORY_BUS_WIDTH_FIELD] = memoryBusWidth;
    device[detail::WINNER_LINE_MEMORY_CLOCK_RATE_FIELD] = memoryClockRate;
    device[detail::WINNER_LINE_SHARED_MEM_PER_BLOCK_FIELD] = sharedMemPerBlock;

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

        // Every DeviceKey field is required: a line missing one (a version-1 line, or a
        // truncated one) must miss, never decode with a zero the writer did not measure.
        DeviceProperties properties;
        properties.gcnArchName
            = deviceField->at(detail::WINNER_LINE_GCN_ARCH_NAME_FIELD).get<std::string>();
        const auto warpSize
            = detail::readNonNegative<int>(*deviceField, detail::WINNER_LINE_WARP_SIZE_FIELD);
        const auto multiProcessorCount = detail::readNonNegative<int>(
            *deviceField, detail::WINNER_LINE_MULTI_PROCESSOR_COUNT_FIELD);
        const auto totalGlobalMem = detail::readNonNegative<std::size_t>(
            *deviceField, detail::WINNER_LINE_TOTAL_GLOBAL_MEM_FIELD);
        const auto memoryBusWidth = detail::readNonNegative<int>(
            *deviceField, detail::WINNER_LINE_MEMORY_BUS_WIDTH_FIELD);
        const auto memoryClockRate = detail::readNonNegative<int>(
            *deviceField, detail::WINNER_LINE_MEMORY_CLOCK_RATE_FIELD);
        const auto sharedMemPerBlock = detail::readNonNegative<std::size_t>(
            *deviceField, detail::WINNER_LINE_SHARED_MEM_PER_BLOCK_FIELD);
        if(!warpSize.has_value() || !multiProcessorCount.has_value() || !totalGlobalMem.has_value()
           || !memoryBusWidth.has_value() || !memoryClockRate.has_value()
           || !sharedMemPerBlock.has_value())
        {
            return std::nullopt;
        }
        properties.warpSize = *warpSize;
        properties.multiProcessorCount = *multiProcessorCount;
        properties.totalGlobalMem = *totalGlobalMem;
        properties.memoryBusWidth = *memoryBusWidth;
        properties.memoryClockRate = *memoryClockRate;
        properties.sharedMemPerBlock = *sharedMemPerBlock;

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
