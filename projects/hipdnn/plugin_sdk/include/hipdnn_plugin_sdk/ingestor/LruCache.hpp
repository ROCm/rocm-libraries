// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <functional>
#include <list>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hipdnn_plugin_sdk::ingestor
{

/// A bounded, thread-safe least-recently-used cache, sized by entry count.
/// @tparam Value Copied in and out; callers hold snapshots, not references.
template <typename Key, typename Value, typename Hash = std::hash<Key>>
class LruCache
{
public:
    using Entry = std::pair<Key, Value>;

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

    /// Merges a batch whose entries are ordered newest-first, under ONE hold of the lock:
    /// a key already present is left alone (whatever is in memory is newer than the batch),
    /// and of several batch entries for one key only the first is taken. Which keys are
    /// absent is decided before anything is inserted, so an entry evicted by the batch
    /// itself can never be refilled by an older value later in that batch -- the failure a
    /// putIfAbsent() loop has once the batch outgrows the capacity.
    ///
    /// The first admitted entry ends up most-recently-used, so when the batch exceeds the
    /// capacity it is the newest entries that stay resident.
    void mergeAbsent(std::vector<Entry> newestFirst)
    {
        const std::lock_guard<std::mutex> lock(_mutex);

        // References into the batch, which does not move while they live: the keys are
        // compared, never stored. equal_to<Key> unwraps the references; equal_to<> would
        // compare the reference_wrappers themselves, which have no operator==.
        // NOLINTNEXTLINE(modernize-use-transparent-functors) - must convert to const Key&
        std::unordered_set<std::reference_wrapper<const Key>, Hash, std::equal_to<Key>> seen;
        std::vector<Entry*> admitted;
        admitted.reserve(newestFirst.size());
        for(auto& entry : newestFirst)
        {
            if(_index.find(entry.first) != _index.end() || !seen.insert(entry.first).second)
            {
                continue;
            }
            admitted.push_back(&entry);
        }

        // Anything past the first `_capacity` admitted entries would be evicted by the newer
        // ones inserted after it, so it is never inserted at all.
        const auto resident = std::min(admitted.size(), _capacity);
        for(auto index = resident; index-- > 0;)
        {
            _order.emplace_front(std::move(*admitted[index]));
            _index[_order.front().first] = _order.begin();
            if(_index.size() > _capacity)
            {
                _index.erase(_order.back().first);
                _order.pop_back();
            }
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
    mutable std::mutex _mutex;
    size_t _capacity;
    std::list<Entry> _order; ///< Most-recently-used first.
    std::unordered_map<Key, typename std::list<Entry>::iterator, Hash> _index;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
