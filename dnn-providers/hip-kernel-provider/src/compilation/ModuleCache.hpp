// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>

namespace hip_kernel_provider::compilation
{

/// @brief Thread-safe CRTP cache for loaded modules.
///
/// Provides getOrLoad(), contains(), and size() backed by a mutex-protected
/// unordered_map.  The Derived class must supply two static methods:
///
///   static std::string makeKey(Args... args);
///   static Value       load(Args... args);
///
/// A load() that returns a falsy value (nullptr, nullopt, etc.) is treated
/// as a failure and is NOT cached.
///
/// @tparam Derived  CRTP subclass that provides makeKey() and load()
/// @tparam Value    Cached value type (must be contextually convertible to bool)
/// @tparam Args     Argument types forwarded to makeKey() and load()
template <typename Derived, typename Value, typename... Args>
class ModuleCache
{
    friend Derived;

public:
    /// Look up or create a cached entry.  On cache miss, calls
    /// Derived::load(args...) and caches the result only if it is truthy.
    Value getOrLoad(Args... args)
    {
        const std::string key = Derived::makeKey(args...);

        const std::lock_guard<std::mutex> lock(_mutex);
        auto it = _entries.find(key);
        if(it != _entries.end())
        {
            return it->second;
        }

        auto loaded = Derived::load(args...);
        if(!loaded)
        {
            return loaded;
        }
        _entries.emplace(key, loaded);
        return loaded;
    }

    /// Check whether a key is present in the cache.
    bool contains(Args... args) const
    {
        const std::string key = Derived::makeKey(args...);
        const std::lock_guard<std::mutex> lock(_mutex);
        return _entries.find(key) != _entries.end();
    }

    /// Return the number of cached entries.
    size_t size() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _entries.size();
    }

    // NOLINTBEGIN(bugprone-crtp-constructor-accessibility)
    ModuleCache(const ModuleCache&) = delete;
    ModuleCache& operator=(const ModuleCache&) = delete;
    ModuleCache(ModuleCache&&) = delete;
    ModuleCache& operator=(ModuleCache&&) = delete;
    // NOLINTEND(bugprone-crtp-constructor-accessibility)

private:
    ModuleCache() = default;
    ~ModuleCache() = default;

    mutable std::mutex _mutex;
    std::unordered_map<std::string, Value> _entries;
};

} // namespace hip_kernel_provider::compilation
