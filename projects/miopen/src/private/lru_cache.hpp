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
#include <unordered_map>
#include <utility>

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

    void Insert(const Key& key, Value value)
    {
        if(Value* existing = Find(key))
        {
            *existing = std::move(value);
            return;
        }
        entries_.emplace_front(key, std::move(value));
        index_.emplace(key, entries_.begin());
        if(entries_.size() > capacity_)
        {
            index_.erase(entries_.back().first);
            entries_.pop_back();
        }
    }

    template <class Predicate>
    void EraseIf(Predicate predicate)
    {
        for(auto it = entries_.begin(); it != entries_.end();)
        {
            if(predicate(it->first))
            {
                index_.erase(it->first);
                it = entries_.erase(it);
            }
            else
            {
                ++it;
            }
        }
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
