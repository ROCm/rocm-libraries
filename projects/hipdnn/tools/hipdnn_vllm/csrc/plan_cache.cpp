// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "plan_cache.h"

#include "stats.h"

namespace hipdnn_vllm
{

std::shared_ptr<PlanEntry> PlanCache::get(const std::string& key)
{
    const std::lock_guard<std::mutex> lock(_mutex);
    auto it = _index.find(key);
    if(it == _index.end())
    {
        return nullptr;
    }
    if(it->second != _lru.begin())
    {
        _lru.splice(_lru.begin(), _lru, it->second);
    }
    return it->second->second;
}

void PlanCache::put(const std::string& key, std::shared_ptr<PlanEntry> entry)
{
    const std::lock_guard<std::mutex> lock(_mutex);
    auto it = _index.find(key);
    if(it != _index.end())
    {
        it->second->second = std::move(entry);
        _lru.splice(_lru.begin(), _lru, it->second);
        return;
    }
    _lru.emplace_front(key, std::move(entry));
    _index.emplace(key, _lru.begin());
    evictLocked();
}

void PlanCache::evictLocked()
{
    while(_lru.size() > _capacity)
    {
        // Entries are shared_ptr: a plan evicted while another thread executes
        // it stays alive until that execute returns.
        _index.erase(_lru.back().first);
        _lru.pop_back();
        Stats::instance().planCacheEviction();
    }
}

void PlanCache::setCapacity(size_t capacity)
{
    const std::lock_guard<std::mutex> lock(_mutex);
    _capacity = capacity == 0 ? 1 : capacity;
    evictLocked();
}

size_t PlanCache::size() const
{
    const std::lock_guard<std::mutex> lock(_mutex);
    return _lru.size();
}

size_t PlanCache::capacity() const
{
    const std::lock_guard<std::mutex> lock(_mutex);
    return _capacity;
}

void PlanCache::clear()
{
    const std::lock_guard<std::mutex> lock(_mutex);
    _index.clear();
    _lru.clear();
}

void PlanCache::pin(const std::shared_ptr<PlanEntry>& entry)
{
    const std::lock_guard<std::mutex> lock(_mutex);
    _pinned.insert(entry);
}

size_t PlanCache::pinnedCount() const
{
    const std::lock_guard<std::mutex> lock(_mutex);
    return _pinned.size();
}

} // namespace hipdnn_vllm
