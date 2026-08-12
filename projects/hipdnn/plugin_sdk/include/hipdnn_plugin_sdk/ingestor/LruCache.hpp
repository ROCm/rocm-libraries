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

/**
 * @brief A bounded, thread-safe least-recently-used cache.
 *
 * Sized by entry count rather than bytes: the ingestor's entries hold descriptor ids
 * and bound field values, not kernels or graphs, so their size is bounded by
 * construction.
 *
 * Eviction costs a rematch, never a wrong answer, so the capacity is a
 * memory/latency tradeoff rather than a correctness one.
 *
 * @tparam Key   Must be hashable by @p Hash and equality-comparable.
 * @tparam Value Copied in and out; callers hold snapshots, not references into the cache.
 * @tparam Hash  Defaults to std::hash<Key>. Composite keys need an explicit hash, since
 *               the standard library provides none for std::pair.
 */
template <typename Key, typename Value, typename Hash = std::hash<Key>>
class LruCache
{
public:
    /// @throws std::invalid_argument if @p capacity is zero, which would make every
    ///         insertion immediately evict itself — always a caller bug, never intent.
    explicit LruCache(size_t capacity)
        : _capacity(capacity)
    {
        if(capacity == 0)
        {
            throw std::invalid_argument("LruCache capacity must be non-zero");
        }
    }

    /// @brief Looks up @p key, marking it most-recently-used on a hit.
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

    /// @brief Inserts or overwrites @p key, marking it most-recently-used and evicting
    ///        the least-recently-used entry if that pushes the cache over capacity.
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
    /// Most-recently-used first. A list so splicing an entry to the front on a hit
    /// does not invalidate the iterators held in _index.
    std::list<Entry> _order;
    std::unordered_map<Key, typename std::list<Entry>::iterator, Hash> _index;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
