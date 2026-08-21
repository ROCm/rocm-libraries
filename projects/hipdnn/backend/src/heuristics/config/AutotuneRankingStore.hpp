// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_data_sdk/utilities/CacheRoot.hpp>
#include <hipdnn_data_sdk/utilities/LineStore.hpp>
#include <hipdnn_data_sdk/utilities/PathSanitizer.hpp>
#include <hipdnn_data_sdk/version.h>

#include <nlohmann/json.hpp>

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

/// Abstract exact-match record store; concatenates key and deviceKey internally.
class IAutotuneRankingStore
{
public:
    virtual ~IAutotuneRankingStore() = default;

    virtual void put(const std::vector<uint8_t>& key,
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
/// (graph, device) key. Every failure fails soft: get() returns nullopt and put() does nothing;
/// nothing here throws.
class FileAutotuneRankingStore : public IAutotuneRankingStore
{
public:
    void put(const std::vector<uint8_t>& key,
             const std::vector<uint8_t>& deviceKey,
             const std::vector<int64_t>& sampledEngineIds,
             const std::vector<int64_t>& order) override
    {
        const std::string combinedKeyHex = combinedKey(key, deviceKey);
        const auto shardPath = shardPathFor(combinedKeyHex);
        if(!shardPath.has_value())
        {
            return;
        }

        auto [shard, openStatus]
            = hipdnn_data_sdk::utilities::openLineStore(*shardPath, detail::shardVersion());
        if(openStatus != hipdnn_data_sdk::utilities::LineStoreStatus::OK || !shard.has_value())
        {
            return;
        }

        if(hipdnn_data_sdk::utilities::lockLineStore(*shard)
           != hipdnn_data_sdk::utilities::LineStoreStatus::OK)
        {
            return;
        }

        // Re-reads under the lock and adopts an existing record for this key instead of appending a
        // duplicate, guarding against two processes racing the same miss.
        const auto [existing, readStatus]
            = hipdnn_data_sdk::utilities::readAllLines(*shard, [&](std::string_view line) {
                  return detail::decodeRecordLine(line, detail::shardVersion());
              });
        if(readStatus == hipdnn_data_sdk::utilities::LineStoreStatus::OK)
        {
            for(const auto& record : existing)
            {
                if(record.combinedKeyHex == combinedKeyHex)
                {
                    hipdnn_data_sdk::utilities::unlockLineStore(*shard);
                    return;
                }
            }
        }

        CachedEntry entry;
        entry.sampledEngineIds = sampledEngineIds;
        entry.order = order;
        hipdnn_data_sdk::utilities::appendLine(*shard,
                                               detail::encodeRecordLine(combinedKeyHex, entry));
        hipdnn_data_sdk::utilities::unlockLineStore(*shard);
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
        const auto shardPath = shardPathFor(combinedKeyHex);
        if(!shardPath.has_value())
        {
            setStatus(RankingLookupStatus::UNAVAILABLE);
            return std::nullopt;
        }

        auto [shard, openStatus]
            = hipdnn_data_sdk::utilities::openLineStore(*shardPath, detail::shardVersion());
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
    static std::string combinedKey(const std::vector<uint8_t>& key,
                                   const std::vector<uint8_t>& deviceKey)
    {
        std::vector<uint8_t> combined;
        combined.reserve(key.size() + deviceKey.size());
        combined.insert(combined.end(), key.begin(), key.end());
        combined.insert(combined.end(), deviceKey.begin(), deviceKey.end());
        return detail::hexEncode(combined);
    }

    /// Resolves the shard path for @p combinedKeyHex, or nullopt if the cache root is unavailable.
    static std::optional<std::filesystem::path> shardPathFor(const std::string& combinedKeyHex)
    {
        const auto root = hipdnn_data_sdk::utilities::cacheRoot();
        if(root.empty())
        {
            return std::nullopt;
        }

        const auto subtree = root / "autotune-rankings" / detail::shardVersion();
        std::error_code failed;
        std::filesystem::create_directories(subtree, failed);
        if(failed || !std::filesystem::is_directory(subtree))
        {
            return std::nullopt;
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
