// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <list>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace hipdnn_plugin_sdk::ingestor
{

/// A bounded, thread-safe least-recently-used cache, sized by entry count.
/// @tparam Value Copied in and out; callers hold snapshots, not references.
template <typename Key, typename Value, typename Hash = std::hash<Key>>
class LruCache
{
public:
    /// @throws std::invalid_argument if @p capacity is zero.
    explicit LruCache(size_t capacity)
        : _capacity(capacity)
    {
        if(capacity == 0)
        {
            throw std::invalid_argument("LruCache capacity must be non-zero");
        }
    }

    /// @return A copy of the cached value, or nullopt on a miss.
    std::optional<Value> get(const Key& key)
    {
        const std::lock_guard<std::mutex> lock(_mutex);

        auto it = _index.find(key);
        if(it == _index.end())
        {
            return std::nullopt;
        }

        _order.splice(_order.begin(), _order, it->second);
        return it->second->second;
    }

    void put(const Key& key, Value value)
    {
        const std::lock_guard<std::mutex> lock(_mutex);

        auto it = _index.find(key);
        if(it != _index.end())
        {
            it->second->second = std::move(value);
            _order.splice(_order.begin(), _order, it->second);
            return;
        }

        _order.emplace_front(key, std::move(value));
        _index[key] = _order.begin();

        if(_index.size() > _capacity)
        {
            _index.erase(_order.back().first);
            _order.pop_back();
        }
    }

    /// Inserts only if absent; use over put() when a racing writer may already have
    /// installed a value strictly better than this one (e.g. unsorted vs. ranked).
    /// @return true if the value was inserted.
    bool putIfAbsent(const Key& key, Value value)
    {
        const std::lock_guard<std::mutex> lock(_mutex);

        if(auto it = _index.find(key); it != _index.end())
        {
            _order.splice(_order.begin(), _order, it->second);
            return false;
        }

        _order.emplace_front(key, std::move(value));
        _index[key] = _order.begin();

        if(_index.size() > _capacity)
        {
            _index.erase(_order.back().first);
            _order.pop_back();
        }
        return true;
    }

    size_t size() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _index.size();
    }

    size_t capacity() const
    {
        return _capacity;
    }

private:
    using Entry = std::pair<Key, Value>;

    mutable std::mutex _mutex;
    size_t _capacity;
    std::list<Entry> _order; ///< Most-recently-used first.
    std::unordered_map<Key, typename std::list<Entry>::iterator, Hash> _index;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
