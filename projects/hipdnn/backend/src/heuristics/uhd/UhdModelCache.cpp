// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "UhdModelCache.hpp"

namespace hipdnn_backend::heuristics::uhd
{

UhdModelCache& UhdModelCache::instance()
{
    static UhdModelCache s_instance;
    return s_instance;
}

std::shared_ptr<IUhdAdapter>
    UhdModelCache::getOrLoad(const CacheKey& key,
                              const std::function<std::unique_ptr<IUhdAdapter>()>& loader)
{
    const std::lock_guard<std::mutex> lock(_mutex);

    const auto it = _cache.find(key);
    if(it != _cache.end())
    {
        return it->second;
    }

    auto adapter = loader();
    if(adapter == nullptr)
    {
        return nullptr;
    }

    const auto shared = std::shared_ptr<IUhdAdapter>(std::move(adapter));
    _cache[key] = shared;
    return shared;
}

bool UhdModelCache::contains(const CacheKey& key) const
{
    const std::lock_guard<std::mutex> lock(_mutex);
    return _cache.find(key) != _cache.end();
}

void UhdModelCache::clear()
{
    const std::lock_guard<std::mutex> lock(_mutex);
    _cache.clear();
}

size_t UhdModelCache::size() const
{
    const std::lock_guard<std::mutex> lock(_mutex);
    return _cache.size();
}

} // namespace hipdnn_backend::heuristics::uhd
