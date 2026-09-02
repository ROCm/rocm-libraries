// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_data_sdk/utilities/CacheRoot.hpp>
#include <hipdnn_data_sdk/utilities/LineStore.hpp>
#include <hipdnn_data_sdk/utilities/PathSanitizer.hpp>
#include <hipdnn_data_sdk/version.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace hipdnn_backend::heuristics::config
{

/// One persisted exact-match autotune record.
struct CachedEntry
{
    std::vector<int64_t> sampledEngineIds;
    std::vector<int64_t> order;
    std::string version;
};

/// Outcome of IAutotuneRankingStore::get(): an ordinary miss versus an unreadable shard.
enum class RankingLookupStatus
{
    HIT,
    MISS,
    UNAVAILABLE,
};

/// Outcome of IAutotuneRankingStore::put().
///
/// UNCHANGED is not a failure: the shard already held this exact ranking, so there was
/// nothing to write. Callers that report a write to the user must distinguish it from
/// WRITTEN, or they claim to have persisted something they did not.
enum class RankingWriteStatus
{
    WRITTEN,
    UNCHANGED,
    UNAVAILABLE,
};

/// Abstract exact-match record store; concatenates key and deviceKey internally.
class IAutotuneRankingStore
{
public:
    virtual ~IAutotuneRankingStore() = default;

    virtual RankingWriteStatus put(const std::vector<uint8_t>& key,
                                   const std::vector<uint8_t>& deviceKey,
                                   const std::vector<int64_t>& sampledEngineIds,
                                   const std::vector<int64_t>& order)
        = 0;

    /// Sets *outStatus to HIT, MISS, or UNAVAILABLE; the return value is nullopt for both.
    virtual std::optional<CachedEntry> get(const std::vector<uint8_t>& key,
                                           const std::vector<uint8_t>& deviceKey,
                                           RankingLookupStatus* outStatus = nullptr) const
        = 0;
};

namespace detail
{

/// Hex-encodes @p bytes for use as the record's key field and the shard path component.
inline std::string hexEncode(const std::vector<uint8_t>& bytes)
{
    static constexpr std::array<char, 16> DIGITS
        = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
    std::string out;
    out.reserve(bytes.size() * 2);
    for(const uint8_t b : bytes)
    {
        out.push_back(DIGITS[(b >> 4) & 0xF]);
        out.push_back(DIGITS[b & 0xF]);
    }
    return out;
}

/// Version string every shard is checked against; shards from a different version are a mismatch.
/// "unknown" (no resolvable git hash) is a valid value.
inline const std::string& shardVersion()
{
    static const std::string s_version = HIPDNN_DATA_SDK_VERSION_STRING;
    return s_version;
}

/// Encodes one record as a single-line JSON object; LineStore itself stays format-agnostic.
inline std::string encodeRecordLine(const std::string& combinedKeyHex, const CachedEntry& entry)
{
    nlohmann::json j;
    j["key"] = combinedKeyHex;
    j["sampledEngineIds"] = entry.sampledEngineIds;
    j["order"] = entry.order;
    return j.dump();
}

struct DecodedRecord
{
    std::string combinedKeyHex;
    CachedEntry entry;
};

/// Declines (nullopt) on any malformed line without throwing, so one bad line never poisons the
/// rest of the shard.
inline std::optional<DecodedRecord> decodeRecordLine(std::string_view line,
                                                     const std::string& version)
{
    nlohmann::json j;
    try
    {
        j = nlohmann::json::parse(line);
    }
    catch(const nlohmann::json::exception&)
    {
        return std::nullopt;
    }

    if(!j.is_object() || !j.contains("key") || !j.contains("sampledEngineIds")
       || !j.contains("order"))
    {
        return std::nullopt;
    }

    try
    {
        DecodedRecord decoded;
        decoded.combinedKeyHex = j.at("key").get<std::string>();
        decoded.entry.sampledEngineIds = j.at("sampledEngineIds").get<std::vector<int64_t>>();
        decoded.entry.order = j.at("order").get<std::vector<int64_t>>();
        decoded.entry.version = version;
        return decoded;
    }
    catch(const nlohmann::json::exception&)
    {
        return std::nullopt;
    }
}

} // namespace detail

/// File-backed exact-match ranking store built on LineStore. On-disk layout:
/// $HIPDNN_CACHE_DIR/autotune-rankings/<data_sdk-version>/<combined-key-hex>.jsonl, one shard per
/// (graph, device) key. Every failure fails soft: get() returns nullopt and put() reports
/// UNAVAILABLE; nothing here throws.
class FileAutotuneRankingStore : public IAutotuneRankingStore
{
public:
    /// Appends @p order under the (key, deviceKey) pair unless the shard already holds that exact
    /// ranking.
    ///
    /// A shard may hold several lines for one key; get() resolves them last-line-wins, so
    /// appending a newer line is how a record is replaced. Without that, a record would be
    /// permanent: once an engine is installed that the stored ranking never measured, the read
    /// path's C\S rule rejects the entry on every lookup, and re-tuning could never repair it.
    ///
    /// @return UNCHANGED if the shard's current ranking for this key already matches, so nothing
    ///     was written; WRITTEN if a line was appended; UNAVAILABLE if the shard could not be
    ///     opened, locked, or read.
    RankingWriteStatus put(const std::vector<uint8_t>& key,
                           const std::vector<uint8_t>& deviceKey,
                           const std::vector<int64_t>& sampledEngineIds,
                           const std::vector<int64_t>& order) override
    {
        const std::string combinedKeyHex = combinedKey(key, deviceKey);
        const auto shardPath = shardPathFor(combinedKeyHex, /*createSubtree=*/true);
        if(!shardPath.has_value())
        {
            return RankingWriteStatus::UNAVAILABLE;
        }

        auto [shard, openStatus]
            = hipdnn_data_sdk::utilities::openLineStore(*shardPath, detail::shardVersion());
        if(openStatus != hipdnn_data_sdk::utilities::LineStoreStatus::OK || !shard.has_value())
        {
            return RankingWriteStatus::UNAVAILABLE;
        }

        if(hipdnn_data_sdk::utilities::lockLineStore(*shard)
           != hipdnn_data_sdk::utilities::LineStoreStatus::OK)
        {
            return RankingWriteStatus::UNAVAILABLE;
        }

        // Read-then-append is one critical section under the shard's own lock: the racing writer
        // may be another process, which _winnerCacheMutex-style in-process locking cannot see.
        const auto [existing, readStatus]
            = hipdnn_data_sdk::utilities::readAllLines(*shard, [&](std::string_view line) {
                  return detail::decodeRecordLine(line, detail::shardVersion());
              });
        if(readStatus != hipdnn_data_sdk::utilities::LineStoreStatus::OK)
        {
            hipdnn_data_sdk::utilities::unlockLineStore(*shard);
            return RankingWriteStatus::UNAVAILABLE;
        }

        // Compare against the LAST line for this key, mirroring get()'s last-line-wins merge.
        // Once a shard can hold a superseded line, the first match is stale, and comparing
        // against it would adopt a record the reader has already replaced.
        const detail::DecodedRecord* current = nullptr;
        for(const auto& record : existing)
        {
            if(record.combinedKeyHex == combinedKeyHex)
            {
                current = &record;
            }
        }

        if(current != nullptr)
        {
            if(current->entry.sampledEngineIds == sampledEngineIds && current->entry.order == order)
            {
                // Byte-identical ranking. This is also the racing-writer case the re-read exists
                // for: two processes that raced the same miss measured the same engines and
                // produced the same order, so the loser has nothing to add.
                hipdnn_data_sdk::utilities::unlockLineStore(*shard);
                return RankingWriteStatus::UNCHANGED;
            }

            if(isStrictSubset(sampledEngineIds, current->entry.sampledEngineIds))
            {
                // A deliberately narrowed sweep (engineIdFilter) measured fewer engines than the
                // stored record covers. Letting it win would replace a usable full-coverage
                // ranking with one that rejects on every later lookup, permanently de-optimising
                // the machine through an otherwise legitimate call.
                hipdnn_data_sdk::utilities::unlockLineStore(*shard);
                return RankingWriteStatus::UNCHANGED;
            }
        }

        CachedEntry entry;
        entry.sampledEngineIds = sampledEngineIds;
        entry.order = order;
        const auto appendStatus = hipdnn_data_sdk::utilities::appendLine(
            *shard, detail::encodeRecordLine(combinedKeyHex, entry));
        hipdnn_data_sdk::utilities::unlockLineStore(*shard);
        return appendStatus == hipdnn_data_sdk::utilities::LineStoreStatus::OK
                   ? RankingWriteStatus::WRITTEN
                   : RankingWriteStatus::UNAVAILABLE;
    }

    std::optional<CachedEntry> get(const std::vector<uint8_t>& key,
                                   const std::vector<uint8_t>& deviceKey,
                                   RankingLookupStatus* outStatus = nullptr) const override
    {
        auto setStatus = [outStatus](RankingLookupStatus status) {
            if(outStatus != nullptr)
            {
                *outStatus = status;
            }
        };

        const std::string combinedKeyHex = combinedKey(key, deviceKey);
        // No subtree creation: a lookup that misses must leave no trace.
        const auto shardPath = shardPathFor(combinedKeyHex, /*createSubtree=*/false);
        if(!shardPath.has_value())
        {
            setStatus(RankingLookupStatus::UNAVAILABLE);
            return std::nullopt;
        }

        auto [shard, openStatus]
            = hipdnn_data_sdk::utilities::openExistingLineStore(*shardPath, detail::shardVersion());
        if(openStatus == hipdnn_data_sdk::utilities::LineStoreStatus::NOT_FOUND)
        {
            // Nothing was ever written under this key: an ordinary miss, and the common
            // case for any graph that has not been swept.
            setStatus(RankingLookupStatus::MISS);
            return std::nullopt;
        }
        if(openStatus != hipdnn_data_sdk::utilities::LineStoreStatus::OK || !shard.has_value())
        {
            setStatus(RankingLookupStatus::UNAVAILABLE);
            return std::nullopt;
        }

        const auto [records, readStatus]
            = hipdnn_data_sdk::utilities::readAllLines(*shard, [&](std::string_view line) {
                  return detail::decodeRecordLine(line, detail::shardVersion());
              });
        if(readStatus != hipdnn_data_sdk::utilities::LineStoreStatus::OK)
        {
            setStatus(RankingLookupStatus::UNAVAILABLE);
            return std::nullopt;
        }

        // Last-line-wins: LineStore guarantees line order, not record identity.
        const detail::DecodedRecord* winner = nullptr;
        for(const auto& record : records)
        {
            if(record.combinedKeyHex == combinedKeyHex)
            {
                winner = &record;
            }
        }

        if(winner == nullptr)
        {
            setStatus(RankingLookupStatus::MISS);
            return std::nullopt;
        }

        setStatus(RankingLookupStatus::HIT);
        return winner->entry;
    }

private:
    /// Is every id in @p candidate also in @p superset, with @p superset strictly larger?
    ///
    /// Order-insensitive: `sampledEngineIds` is a set in meaning, and the sweep that produced it
    /// may enumerate engines in any order. Sized for a handful of engines, so the quadratic scan
    /// is cheaper than sorting copies.
    static bool isStrictSubset(const std::vector<int64_t>& candidate,
                               const std::vector<int64_t>& superset)
    {
        if(candidate.size() >= superset.size())
        {
            return false;
        }
        return std::all_of(candidate.begin(), candidate.end(), [&superset](int64_t id) {
            return std::find(superset.begin(), superset.end(), id) != superset.end();
        });
    }

    static std::string combinedKey(const std::vector<uint8_t>& key,
                                   const std::vector<uint8_t>& deviceKey)
    {
        std::vector<uint8_t> combined;
        combined.reserve(key.size() + deviceKey.size());
        combined.insert(combined.end(), key.begin(), key.end());
        combined.insert(combined.end(), deviceKey.begin(), deviceKey.end());
        return detail::hexEncode(combined);
    }

    /// Resolves the shard path for @p combinedKeyHex, or nullopt if the cache root is
    /// unavailable.
    ///
    /// @param createSubtree Create the versioned subtree when absent. True for put(),
    ///     which is about to write into it; false for get(), where an absent subtree
    ///     already means nothing has been written.
    static std::optional<std::filesystem::path> shardPathFor(const std::string& combinedKeyHex,
                                                             bool createSubtree)
    {
        const auto root = hipdnn_data_sdk::utilities::cacheRoot();
        if(root.empty())
        {
            return std::nullopt;
        }

        const auto subtree = root / "autotune-rankings" / detail::shardVersion();
        if(createSubtree)
        {
            std::error_code failed;
            std::filesystem::create_directories(subtree, failed);
            if(failed || !std::filesystem::is_directory(subtree))
            {
                return std::nullopt;
            }
        }

        return subtree / (hipdnn_data_sdk::utilities::sanitizeForPath(combinedKeyHex) + ".jsonl");
    }
};

/// Process-local accessor for the store shared by the read and write paths; stateless on disk.
inline IAutotuneRankingStore& exactCacheStore()
{
    static FileAutotuneRankingStore s_store;
    return s_store;
}

} // namespace hipdnn_backend::heuristics::config
