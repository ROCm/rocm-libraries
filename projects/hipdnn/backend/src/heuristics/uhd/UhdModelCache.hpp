// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "adapters/IUhdAdapter.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace hipdnn_backend::heuristics::uhd
{

/// @brief Thread-safe cache for loaded UHD model adapters.
///
/// Adapters are keyed by (engine_id, device_id, uhd_id) and loaded lazily on
/// first access. The cache lives for the lifetime of the process.
class UhdModelCache
{
public:
    /// Get the singleton cache instance.
    static UhdModelCache& instance();

    /// Cache key combining engine, device, and UHD identifiers.
    struct CacheKey
    {
        int64_t engineId;
        int64_t deviceId;
        std::string uhdId;

        bool operator==(const CacheKey& other) const
        {
            return engineId == other.engineId && deviceId == other.deviceId && uhdId == other.uhdId;
        }
    };

    struct CacheKeyHash
    {
        size_t operator()(const CacheKey& key) const
        {
            size_t h1 = std::hash<int64_t>{}(key.engineId);
            size_t h2 = std::hash<int64_t>{}(key.deviceId);
            size_t h3 = std::hash<std::string>{}(key.uhdId);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };

    /// Get or load an adapter for the given key.
    /// @param key Cache key.
    /// @param loader Function to load the adapter if not cached.
    /// @returns Adapter pointer (may be null if loading failed).
    std::shared_ptr<IUhdAdapter> getOrLoad(const CacheKey& key,
                                            std::function<std::unique_ptr<IUhdAdapter>()> loader);

    /// Check if an adapter is cached for the given key.
    bool contains(const CacheKey& key) const;

    /// Clear the entire cache.
    void clear();

    /// Get the number of cached adapters.
    size_t size() const;

private:
    UhdModelCache() = default;
    UhdModelCache(const UhdModelCache&) = delete;
    UhdModelCache& operator=(const UhdModelCache&) = delete;

    mutable std::mutex _mutex;
    std::unordered_map<CacheKey, std::shared_ptr<IUhdAdapter>, CacheKeyHash> _cache;
};

} // namespace hipdnn_backend::heuristics::uhd
