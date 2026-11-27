// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Fixed-capacity map that evicts the least recently used entry once full. Not
// thread-safe; callers hold their own lock.
//
// Compiled only into the public wrapper library; never installed.
#pragma once

#include <cstddef>
#include <functional>
#include <list>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace miopen {
namespace wrapper {

template <class Key, class Value, class Hash = std::hash<Key>>
class LruCache
{
public:
    explicit LruCache(std::size_t capacity) : capacity_(capacity) {}

    // Returns nullptr on a miss. A hit becomes the most recently used entry.
    Value* Find(const Key& key)
    {
        auto found = index_.find(key);
        if(found == index_.end())
            return nullptr;
        entries_.splice(entries_.begin(), entries_, found->second);
        return &found->second->second;
    }

    // Returns the value this call displaced instead of destroying it, so the caller can let it
    // go after releasing its lock: the evicted oldest entry when the cache was full, or the old
    // value when `key` was already cached. Empty when nothing was displaced.
    std::optional<Value> Insert(const Key& key, Value value)
    {
        if(Value* existing = Find(key))
        {
            std::optional<Value> replaced(std::move(*existing));
            *existing = std::move(value);
            return replaced;
        }
        entries_.emplace_front(key, std::move(value));
        index_.emplace(key, entries_.begin());
        if(entries_.size() <= capacity_)
            return std::nullopt;
        std::optional<Value> evicted(std::move(entries_.back().second));
        index_.erase(entries_.back().first);
        entries_.pop_back();
        return evicted;
    }

    // Hands the removed values back instead of destroying them, so a caller can
    // let them go after releasing its lock.
    template <class Predicate>
    std::vector<Value> EraseIf(Predicate predicate)
    {
        std::vector<Value> removed;
        for(auto it = entries_.begin(); it != entries_.end();)
        {
            if(predicate(it->first))
            {
                removed.push_back(std::move(it->second));
                index_.erase(it->first);
                it = entries_.erase(it);
            }
            else
            {
                ++it;
            }
        }
        return removed;
    }

    std::size_t Size() const { return entries_.size(); }

private:
    using Entries = std::list<std::pair<Key, Value>>;

    std::size_t capacity_;
    Entries entries_;
    std::unordered_map<Key, typename Entries::iterator, Hash> index_;
};

} // namespace wrapper
} // namespace miopen
